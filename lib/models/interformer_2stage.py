from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import copy
import math
from typing import Optional, List
from models.position_embedding import build_position_encoding
from utils.utils import get_valid_output
import models


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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

class UpConv(nn.Module): 
    """(convolution => [BN] => ReLU) * 2""" 
    def __init__(self, in_channels, out_channels, scale_factor, mid_channels=None): 
        super().__init__() 
        if not mid_channels: 
            mid_channels = out_channels 
            
        self.fuse_layers = self._make_fuse_layers(in_channels, scale_factor)
        self.double_conv = nn.Sequential( 
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True) 
        ) 
        
    def _make_fuse_layers(self, d_model, scale_factor):
        fuse_layer = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                1, 1, 0, bias=False
            ),
            nn.BatchNorm2d(d_model),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return fuse_layer
    
    def forward(self, x): 
        x = self.fuse_layers(x)
        x = self.double_conv(x) 
        return x

class InterFormer(nn.Module):

    def __init__(self, cfg, singleformer, **kwargs):
        super(InterFormer, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.singleformer = singleformer
        self.singleformer_fix = cfg.MODEL.SINGLEFORMER_FIX

        self.trans_size = cfg.MODEL.TRANS_SIZE
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        d_model = cfg.MODEL.DIM_MODEL
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        multi_encoder_layers_num = cfg.MODEL.ENCODER_MULTI_LAYERS
        n_head = cfg.MODEL.N_HEAD
        self.use_multi_pos = cfg.MODEL.USE_MULTI_POS
        self.use_domain_trans=cfg.MODEL.DOMAIN_TRANS
        self.inter_supervision = cfg.MODEL.INTER_SUPERVISION
        self.upsample_type = cfg.MODEL.UPSAMPLE_TYPE

        self.multi_position_mode=cfg.MODEL.MULTI_POS_EMBEDDING
        self.multi_position_dim = cfg.MODEL.MULTI_POS_EMBEDDING_DIM
        
        self.multi_position_embedding = build_position_encoding(self.trans_size, d_model, mode=self.multi_position_mode, vec_dim=self.multi_position_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            activation='relu')

        self.multi_global_encoder = TransformerEncoder(
            encoder_layer, multi_encoder_layers_num)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        
        if self.upsample_type == 'upconv':
            self.upsample_conv = UpConv(d_model, d_model, self.heatmap_size[0] // self.trans_size[1])
        elif self.upsample_type == 'multiplex':
            self.deconv_layers = self._make_deconv_layer(
                extra.NUM_DECONV_LAYERS,   # 1
                extra.NUM_DECONV_FILTERS,  # [d_model]
                extra.NUM_DECONV_KERNELS,  # [4]
            )
        elif self.upsample_type == 'deconv':
            self.deconv_layers1 = self._make_deconv_layer(
                extra.NUM_DECONV_LAYERS,   # 1
                extra.NUM_DECONV_FILTERS,  # [d_model]
                extra.NUM_DECONV_KERNELS,  # [4]
            )
            
            self.deconv_layers2 = self._make_deconv_layer(
                extra.NUM_DECONV_LAYERS,   # 1
                extra.NUM_DECONV_FILTERS,  # [d_model]
                extra.NUM_DECONV_KERNELS,  # [4]
            )
            
            self.deconv_layers3 = self._make_deconv_layer(
                extra.NUM_DECONV_LAYERS,   # 1
                extra.NUM_DECONV_FILTERS,  # [d_model]
                extra.NUM_DECONV_KERNELS,  # [4]
            )

        self.final_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )
        if self.use_domain_trans:
            self.domain_trans_1 = nn.Conv2d(in_channels=d_model, out_channels=d_model,kernel_size=1,stride=1,padding=0,bias=True)
            self.domain_trans_2 = nn.Conv2d(in_channels=d_model, out_channels=d_model,kernel_size=1,stride=1,padding=0,bias=True)

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

    def flatten_input(self, src, mask, pos):
        src = src.permute(0, 2, 1, 3, 4).flatten(2).permute(2, 0, 1)    # [n*h*w, B, C]
        if pos != None:
            pos = pos.permute(0, 2, 1, 3, 4).flatten(2).permute(2, 0, 1)    # [n*h*w, B, C]
        if mask != None:
            mask = mask.flatten(1)
        return src, mask, pos

    def pool_or_deconv(self, x, mod, type):
        rate = int(math.log(mod, 2))
        if type == 'pool':
            for _ in range(rate):
                x = self.maxpool(x)
        elif type == 'upsample':
            if self.upsample_type == 'multiplex':
                for _ in range(rate):
                    x = self.deconv_layers(x)
            elif self.upsample_type == 'deconv':
                for i in range(rate):
                    x = eval('self.deconv_layers'+str(i + 1))(x)
            elif self.upsample_type == 'upconv':
                x = self.upsample_conv(x)
        return x

    def forward(self, x, muti_pos_mask, length):
        outputs = {'single': None, 'multi': None}
        device = x.device
        bs = len(length)
        # x: bs,96,64,48 
        # outputs: bs,17,64,48
        x, outputs['single'] = self.singleformer(x)
        single_res = x
        x = self.pool_or_deconv(x, x.shape[-1] // self.trans_size[-1], 'pool')

        _, c, h, w = x.shape
        x = self.padding_tensor(x, length)
        mask = self.get_mask(length, (h, w))
        mask = mask.to(device)
        
        if self.use_multi_pos:
            muti_pos_mask = self.padding_tensor(muti_pos_mask, length, Learnable=False)
            multi_pos = self.multi_position_embedding(muti_pos_mask)             # [bs,N, 96, 16, 12]
        else:
            multi_pos = None
            
        x, mask, multi_pos = self.flatten_input(x, mask, multi_pos)
            
        x = self.multi_global_encoder(x, src_key_padding_mask=mask, pos=multi_pos)    # [L, bs, c]
        x = x.permute(1, 2, 0).contiguous().view(bs, c, max(length), h, w)
        x = x.permute(0, 2, 1, 3, 4).reshape(bs*max(length), c, h, w)
        x = get_valid_output(x, length)

        x = self.pool_or_deconv(x, single_res.shape[-1] // x.shape[-1], 'upsample')
        
        if self.use_domain_trans:
            x = self.domain_trans_1(single_res) + self.domain_trans_2(x)
        else:
            x = single_res + x
        outputs['multi'] = self.final_layer(x)

        # 若frozen单人阶段，则只返回多人；否则返回全部
        if self.inter_supervision and self.singleformer_fix == False:
            return outputs
        else:
            return outputs['multi']


def get_pose_net(cfg, is_train, **kwargs):

    singleformer = eval('models.'+cfg.MODEL.SINGLEFORMER+'.get_pose_net')(
        cfg, is_train, cfg.MODEL.SINGLE_MODEL, cfg.MODEL.END2END
    )

    model = InterFormer(cfg, singleformer, **kwargs)

    return model