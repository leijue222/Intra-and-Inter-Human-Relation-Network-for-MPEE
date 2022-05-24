# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict

import copy
from typing import Optional, List
from models.position_embedding import build_position_encoding
from utils.utils import get_valid_output


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask,  pos=pos,
                               src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class InterFormer(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(InterFormer, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, multi_scale_output=True)

        self.trans_size = cfg.MODEL.TRANS_SIZE
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        d_model = cfg.MODEL.DIM_MODEL
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        single_encoder_layers_num = cfg.MODEL.ENCODER_SINGLE_LAYERS
        muti_encoder_layers_num = cfg.MODEL.ENCODER_MUTI_LAYERS

        n_head = cfg.MODEL.N_HEAD
        single_pos_embedding_type = cfg.MODEL.SINGLE_POS_EMBEDDING
        w, h = cfg.MODEL.IMAGE_SIZE
        self.res_layer = cfg.MODEL.HRNET_RES_LAYER
        self.use_domain_trans=cfg.MODEL.DOMAIN_TRANS
        
        multi_pos_embedding = cfg.MODEL.MULTI_POS_EMBEDDING


        w, h = w // 2**self.res_layer, h // 2**self.res_layer
        # 预训练模型命名错误，Load模型Miss key的话把multi_position_embedding改成muti_position_embedding即可
        # self.multi_position_embedding = build_position_encoding(self.trans_size, d_model)
        self.multi_position_embedding = build_position_encoding(self.trans_size, d_model, mode=multi_pos_embedding)

        self.reduce = nn.Conv2d(pre_stage_channels[self.res_layer], d_model, 1, bias=False)
        self._make_position_embedding(w, h, d_model, single_pos_embedding_type)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            activation='relu')

        self.single_global_encoder = TransformerEncoder(
            encoder_layer, single_encoder_layers_num)

        self.multi_global_encoder = TransformerEncoder(
            encoder_layer, muti_encoder_layers_num)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,   # 1
            extra.NUM_DECONV_FILTERS,  # [d_model]
            extra.NUM_DECONV_KERNELS,  # [4]
        )

        self.final_layer_single = nn.Conv2d(
            in_channels=d_model,
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )
        
        self.final_layer_multi = nn.Conv2d(
            in_channels=d_model,
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

        self.domain_trans_1 = nn.Conv2d(in_channels=d_model, out_channels=d_model,kernel_size=1,stride=1,padding=0,bias=True)
        self.domain_trans_2 = nn.Conv2d(in_channels=d_model, out_channels=d_model,kernel_size=1,stride=1,padding=0,bias=True)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.single_pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 4
                self.pe_w = w // 4
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.single_pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.single_pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        # dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats) # torch 1.7版本适用
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def deal_by_backbone(self, x):
        # x [N, 3, 256, 192]
        x = self.conv1(x)    # [N, 64, 128, 96]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)    # [N, 64, 64, 48]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)   # [N, 256, 64, 48]

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)  # [N, 48, 64, 48]

        # self.res_layer: [N, 96, 64, 48]    0: 64*48    1: 32*24    2: 16*12
        x = self.reduce(y_list[self.res_layer])
        return x

    def get_mask(self, length, shape):
        N = max(length)
        mask_list = []
        for n in length:
            valid = [torch.zeros(shape, dtype=torch.bool) for _ in range(n)]
            mask_item = torch.stack(valid, dim=0)
            if n < N:
                invalid = [torch.ones(shape, dtype=torch.bool) for _ in range(N-n)]
                invalid = torch.stack(invalid, dim=0)
                mask_item = torch.cat((mask_item, invalid), dim=0)
            mask_list.append(mask_item)

        mask = torch.stack(mask_list, dim=0)
        return mask

    def padding_tensor(self, tensor, length, Learnable=False):
        device = tensor.device
        N = max(length)
        res_list = []

        tensor_split = tensor.split(length, dim=0)

        for valid_item in tensor_split:
            n = len(valid_item)
            # print('n={}, N={}'.format(n, N))
            if n < N:   # need padding
                padding_unit = torch.zeros(tuple(valid_item.shape[-3:]), dtype=torch.float32)
                padding_list = [padding_unit for _ in range(N-n)]
                padding_list = torch.stack(padding_list, dim=0)
                padding_list = padding_list.to(device)
                valid_item = torch.cat((valid_item, padding_list), dim=0)
            res_list.append(valid_item)

        res = torch.stack(res_list, dim=0)
        if Learnable:
            res = nn.Parameter(res)
        return res

    def flatten_input(self, src, pos, mask):
        src = src.permute(0, 2, 1, 3, 4).flatten(2).permute(2, 0, 1)    # [n*h*w, B, C]
        pos = pos.permute(0, 2, 1, 3, 4).flatten(2).permute(2, 0, 1)    # [n*h*w, B, C]
        # mask show be [B, n*h*w]
        mask = mask.flatten(1)

        return src, pos, mask

    def pool_or_deconv(self, x, mod, type):
        rate = int(math.log(mod, 2))
        if type == 'pool':
            for _ in range(rate):
                x = self.maxpool(x)
        elif type == 'deconv':
            for _ in range(rate):
                x = self.deconv_layers(x)
        return x

    def forward(self, x, muti_pos_mask, length):
        outputs = {'single': None, 'multi': None}
        # x [sum(length), 3, 256, 192]
        device = x.device
        bs = len(length)
        # [sum(length), 96, 64, 48]
        x = self.deal_by_backbone(x)
        _, c, h, w = x.shape

        # single Transformer
        x = x.flatten(2).permute(2, 0, 1)
        x = self.single_global_encoder(x, pos=self.single_pos_embedding)
        x = x.permute(1, 2, 0).contiguous().view(sum(length), c, h, w)

        outputs['single'] = self.pool_or_deconv(x, self.heatmap_size[0] // x.shape[-1], 'deconv')

        single_res = x
        x = self.pool_or_deconv(x, x.shape[-1] // self.trans_size[-1], 'pool')

        _, c, h, w = x.shape

        # Multi Transformer
        # [bs, n, c, h, w] sum(length) split into bs, n
        x = self.padding_tensor(x, length)
        mask = self.get_mask(length, (h, w))
        mask = mask.to(device)

        multi_pos = self.padding_tensor(muti_pos_mask, length, Learnable=False)
        multi_pos = self.multi_position_embedding(multi_pos)             # [bs,N, 96, 16, 12]
        x, multi_pos, mask = self.flatten_input(x, multi_pos, mask)
        
        if self.use_multi_pos == False:
            multi_pos = None
            
        x = self.multi_global_encoder(
            x, src_key_padding_mask=mask, pos=multi_pos)    # [L, bs, c]
        x = x.permute(1, 2, 0).contiguous().view(bs, c, max(length), h, w)
        x = x.permute(0, 2, 1, 3, 4).reshape(bs*max(length), c, h, w)
        x = get_valid_output(x, length)

        x = self.pool_or_deconv(x, single_res.shape[-1] // x.shape[-1], 'deconv')
        if self.use_domain_trans:
            x = self.domain_trans_1(single_res) + self.domain_trans_2(x)
        else:
            x = single_res + x
        x = self.pool_or_deconv(x, self.heatmap_size[0] // x.shape[-1], 'deconv')

        outputs['single'] = self.final_layer_single(outputs['single'])
        outputs['multi'] = self.final_layer_multi(x)
        return outputs

    def init_weights(self, pretrained='', fixed=False, print_load_info=False):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, map_location='cpu')
            logger.info('=> loading pretrained model {}'.format(pretrained))

            existing_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers and name in self.state_dict()\
                   or self.pretrained_layers[0] == '*':
                    if fixed:
                        m.requires_grad_(False)
                    existing_state_dict[name] = m
                    if print_load_info:
                        print(":: {} is loaded from {}".format(name, pretrained))
            self.load_state_dict(existing_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = InterFormer(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'], cfg['MODEL']['BACKBONE_FIX'])

    return model
