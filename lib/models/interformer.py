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
from models.attention import get_encoder
from models.backbone import build_backbone

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class UpConv(nn.Module): 
    """(convolution => [BN] => ReLU) * 2""" 
    def __init__(self, cfg): 
        super().__init__() 
            
        in_channels = cfg.MODEL.DIM_MODEL
        mid_channels = cfg.MODEL.DIM_MODEL
        out_channels = cfg.MODEL.DIM_MODEL
        
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.trans_size = cfg.MODEL.TRANS_SIZE
        
        scale_factor = self.heatmap_size[0] // self.trans_size[1]
            
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


class DeConv(nn.Module):
    def __init__(self, cfg): 
        super().__init__() 
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.trans_size = cfg.MODEL.TRANS_SIZE
        
        mod = self.heatmap_size[0] // self.trans_size[1]
        self.layer_num = int(math.log(mod, 2))
        
        self.deconv_layers = nn.ModuleList(self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,   # 1
            extra.NUM_DECONV_FILTERS,  # [d_model]
            extra.NUM_DECONV_KERNELS,  # [4]
        ) for _ in range(self.layer_num))
             
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
    
    def forward(self, x):
        for layer in self.deconv_layers:
            x = layer(x)
        return x
        

class InterFormer(nn.Module):

    def __init__(self, cfg, is_train, **kwargs):
        super(InterFormer, self).__init__()
        extra = cfg.MODEL.EXTRA
        
        self.have_singleformer = cfg.MODEL.SINGLEFORMER
        
        if self.have_singleformer:
            self.singleformer = eval('models.'+cfg.MODEL.SINGLEFORMER+'.get_pose_net')(
                cfg, is_train, cfg.MODEL.SINGLE_MODEL, cfg.MODEL.END2END
            )
        else:
            self.backbone = build_backbone(cfg)
            
        self.singleformer_fix = cfg.MODEL.SINGLEFORMER_FIX
        self.trans_size = cfg.MODEL.TRANS_SIZE
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        d_model = cfg.MODEL.DIM_MODEL
        multi_encoder_layers_num = cfg.MODEL.ENCODER_MULTI_LAYERS
        self.use_multi_pos = cfg.MODEL.USE_MULTI_POS
        self.inter_supervision = cfg.MODEL.INTER_SUPERVISION
        self.upsample_type = cfg.MODEL.UPSAMPLE_TYPE
        self.multi_position_mode=cfg.MODEL.MULTI_POS_EMBEDDING
        self.multi_position_dim = cfg.MODEL.MULTI_POS_EMBEDDING_DIM
        self.multi_position_embedding = build_position_encoding(self.trans_size, d_model, mode=self.multi_position_mode, vec_dim=self.multi_position_dim)

        if self.multi_position_mode == 'cat_vec' and self.use_multi_pos:
            self.fc = nn.Conv2d(in_channels=d_model+self.multi_position_dim,out_channels=d_model,kernel_size=1)
            
        self.multi_global_encoder = get_encoder(cfg, multi_encoder_layers_num)
       
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if self.upsample_type == 'upconv':
            self.upsample_layer = UpConv(cfg)
        elif self.upsample_type == 'deconv':
            self.upsample_layer = DeConv(cfg)
        else:   # multiplex 反卷积复用
            self.deconv_with_bias = extra.DECONV_WITH_BIAS
            self.deconv_layers = self._make_deconv_layer(
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
        
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        def get_deconv_cfg(deconv_kernel):
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
    
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = get_deconv_cfg(num_kernels[i])

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

    def max_pool(self, x, mod,):
        rate = int(math.log(mod, 2))
        for _ in range(rate):
            x = self.maxpool(x)
        return x
    
    def generate_mask(self, x, length):
        device = x.device
        _, _, h, w = x.shape
        x = self.padding_tensor(x, length)
        mask = self.get_mask(length, (h, w))
        mask = mask.to(device)
        return x, mask
    
    def get_multi_position(self, multi_pos_mask, length):
        if self.use_multi_pos:
            multi_pos_mask = self.padding_tensor(multi_pos_mask, length, Learnable=False)
            multi_pos = self.multi_position_embedding(multi_pos_mask)             # [bs,N, 96, 16, 12]
        else:
            multi_pos = None
        return multi_pos

    def forward(self, x, multi_pos_mask, length):
        outputs = {'single': None, 'multi': None}
        
        # bs = len(length)
        
        if self.have_singleformer:
            x, outputs['single'] = self.singleformer(x)
            single_res = x
            x = self.max_pool(x, x.shape[-1] // self.trans_size[-1])
        else:
            x = self.backbone(x)
            
        x, mask = self.generate_mask(x, length)
        multi_pos = self.get_multi_position(multi_pos_mask, length)
        if self.multi_position_mode == 'cat_vec':
            if self.use_multi_pos:
                x = torch.cat([x,multi_pos],dim=2) # bs,n,c,h,w
            x = self.multi_global_encoder(x, key_padding_mask=mask, pos=None)    # [L, bs, c]
            # bs,c,h,w
            if self.use_multi_pos:
                x = self.fc(x)
        else:
            x = self.multi_global_encoder(x, key_padding_mask=mask, pos=multi_pos)    # [L, bs, c]
       
        x = get_valid_output(x, length)

        if self.upsample_type != 'multiplex':
            x = self.upsample_layer(x)
        else:
            x = self.deconv_layers(x)
            x = self.deconv_layers(x)
        
        if self.have_singleformer:
            x = single_res + x      # Residual
            
        outputs['multi'] = self.final_layer(x)

        # Determine whether intermediate supervision is needed
        if self.inter_supervision and self.have_singleformer and self.singleformer_fix == False:
            return outputs
        else:
            return outputs['multi']


def get_pose_net(cfg, is_train, **kwargs):

    model = InterFormer(cfg, is_train, **kwargs)

    return model