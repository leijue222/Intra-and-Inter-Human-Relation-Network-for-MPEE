from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import copy
import math
import warnings
from torch._jit_internal import Optional, Tuple
from torch.overrides import has_torch_function, handle_torch_function
from torch.nn.functional import linear, pad, softmax, dropout
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from functools import partial
import logging
logger = logging.getLogger(__name__)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayerDefault(nn.Module):
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
                     key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=key_padding_mask)[0]
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
                    key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=key_padding_mask)[0]
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
                key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, key_padding_mask, pos)
        return self.forward_post(src, src_mask, key_padding_mask, pos)


class TransformerEncoderDefault(nn.Module):
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
                
    def flatten_input(self, src, mask, pos):
        src = src.permute(0, 2, 1, 3, 4).flatten(2).permute(2, 0, 1)    # [n*h*w, B, C]
        if pos != None and len(pos.size()) == 5:
            pos = pos.permute(0, 2, 1, 3, 4).flatten(2).permute(2, 0, 1)    # [n*h*w, B, C]
        if mask != None:
            mask = mask.flatten(1)
        return src, mask, pos
    
    def flatten_back(self, x, B, N, C, H, W):
        x = x.permute(1, 2, 0).contiguous().view(B, C, N, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B*N, C, H, W)
        return x
        
    def forward(self, src,
                mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        B, N, C, H, W = src.shape
        src, key_padding_mask, pos = self.flatten_input(src, key_padding_mask, pos)
        
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        key_padding_mask=key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask,  pos=pos,
                               key_padding_mask=key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        output = self.flatten_back(output, B, N, C, H, W)
        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


class MultiheadAttention(nn.Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0.0,bias=True, add_bias_kv=False, 
                 add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.in_proj_bias = None
        self.in_proj_weight = None
        self.bias_k = self.bias_v = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.add_zero_attn = add_zero_attn

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, 
                attn_mask=None, residual_attn=None, pos: Optional[Tensor] = None):
        query = key = self.with_pos_embed(query, pos)
        print("\n\n###################\nMultiheadAttention 被执行了!!!\n###################\n\n")
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                out_dim=self.vdim,
                residual_attn=residual_attn,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                out_dim=self.vdim,
                residual_attn=residual_attn,
            )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        out_dim: Optional[Tensor] = None,
        residual_attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not torch.jit.is_scripting():
            tens_ops = (
                query,
                key,
                value,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                out_proj_weight,
                out_proj_bias,
            )
            if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
                tens_ops
            ):
                return handle_torch_function(
                    multi_head_attention_forward,
                    tens_ops,
                    query,
                    key,
                    value,
                    embed_dim_to_check,
                    num_heads,
                    in_proj_weight,
                    in_proj_bias,
                    bias_k,
                    bias_v,
                    add_zero_attn,
                    dropout_p,
                    out_proj_weight,
                    out_proj_bias,
                    training=training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=use_separate_proj_weight,
                    q_proj_weight=q_proj_weight,
                    k_proj_weight=k_proj_weight,
                    v_proj_weight=v_proj_weight,
                    static_k=static_k,
                    static_v=static_v,
                )
        tgt_len, bsz, embed_dim = query.size()
        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        v_head_dim = out_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query) * scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                dim=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        """
        Attention weight for the invalid region is -inf
        """
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        if residual_attn is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights += residual_attn.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
        )
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output


class MHA_(MultiheadAttention):
    """ "Multihead Attention with extra flags on the q/k/v and out projections."""

    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, *args, rpe=False, window_size=7, **kwargs):
        super(MHA_, self).__init__(*args, **kwargs)

        self.rpe = rpe
        if rpe:
            self.window_size = [window_size] * 2
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                    self.num_heads,
                )
            )  # 2*Wh-1 * 2*Ww-1, nH
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None,
        do_qkv_proj=True, do_out_proj=True, rpe=True, pos: Optional[Tensor] = None):
        query = key = self.with_pos_embed(query, pos)
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        out_dim: Optional[Tensor] = None,
        do_qkv_proj: bool = True,
        do_out_proj: bool = True,
        rpe=True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not torch.jit.is_scripting():
            tens_ops = (
                query,
                key,
                value,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                out_proj_weight,
                out_proj_bias,
            )
            if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
                tens_ops
            ):
                return handle_torch_function(
                    multi_head_attention_forward,
                    tens_ops,
                    query,
                    key,
                    value,
                    embed_dim_to_check,
                    num_heads,
                    in_proj_weight,
                    in_proj_bias,
                    bias_k,
                    bias_v,
                    add_zero_attn,
                    dropout_p,
                    out_proj_weight,
                    out_proj_bias,
                    training=training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=use_separate_proj_weight,
                    q_proj_weight=q_proj_weight,
                    k_proj_weight=k_proj_weight,
                    v_proj_weight=v_proj_weight,
                    static_k=static_k,
                    static_v=static_v,
                )
        tgt_len, bsz, embed_dim = query.size()
        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        v_head_dim = out_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        # whether or not use the original query/key/value
        q = self.q_proj(query) * scaling if do_qkv_proj else query
        k = self.k_proj(key) if do_qkv_proj else key
        v = self.v_proj(value) if do_qkv_proj else value

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                dim=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        """
        Add relative position embedding
        """
        if self.rpe and rpe:
            # NOTE: for simplicity, we assume the src_len == tgt_len == window_size**2 here
            # print('src, tar, window', src_len, tgt_len, self.window_size[0], self.window_size[1])
            # assert src_len == self.window_size[0] * self.window_size[1] \
            #                   and tgt_len == self.window_size[0] * self.window_size[1], \
            #                   f"src{src_len}, tgt{tgt_len}, window{self.window_size[0]}"
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            # HELLO!!!!!
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )  # + relative_position_bias.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        """
        Attention weight for the invalid region is -inf
        """
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
        )
        if do_out_proj:
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, q, k, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, q, k  # additionaly return the query and key


class PadBlock(object):
    """ "Make the size of feature map divisible by local group size."""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def pad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
            return F.pad(
                x,
                (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return x

    def depad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
            return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :]
        return x


class LocalPermuteModule(object):
    """ "Permute the feature map to gather pixels in local groups, and the reverse permutation"""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def permute(self, x, b, p, size):
        # x.view(B, P, H, W, C).view(B, P*H*W, C)
        n, h, w, c = size   # n = B*P
        
        # b, (h/7)*7, (w/7)*7, c --> (7*7), b*(h/7)*(w/7), c
        return rearrange(
            x,
            "b (qh ph) (qw pw) c -> (ph pw) (b p qh qw) c",
            b=b,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )

    def rev_permute(self, x, b, p, size):
        n, h, w, c = size
        return rearrange(
            x,
            "(ph pw) (b qh qw) c -> b (qh ph) (qw pw) c",
            b=b,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )


class InterlacedPoolAttention(nn.Module):
    r"""interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, embed_dim, num_heads, window_size=7, rpe=True, **kwargs):
        super(InterlacedPoolAttention, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.with_rpe = rpe

        self.attn = MHA_(
            embed_dim, num_heads, rpe=rpe, window_size=window_size, **kwargs
        )
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)

    def forward(self, x,
                key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # 注释切片分块操作
        # B, N, C = x.shape           #B*P, H*W, C 
        # attention
        # pad 
        # x_pad = self.pad_helper.pad_if_needed(x, x.size())
        # permute 
        # x_permute = self.permute_helper.permute(x_pad, B, P, x_pad.size())
        # reshape
        # x_permute = x.view(B, P, H, W, C).view(B, P*H*W, C)
        # attention
        out, _, _ = self.attn(
            x, x, x, rpe=self.with_rpe, 
            key_padding_mask=key_padding_mask, pos=pos
        )
        # reverse permutation
        # out = self.permute_helper.rev_permute(out, B, P, x_pad.size())
        # de-pad, pooling with `ceil_mode=True` will do implicit padding, so we need to remove it, too
        # out = self.pad_helper.depad_if_needed(out, x.size())
        # return out.reshape(B, N, C)
        return out


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "drop_prob={}".format(self.drop_prob)


class GeneralTransformerBlock(nn.Module):

    def __init__(
        self,
        inplanes,
        num_heads,
        window_size=7,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dim = inplanes
        self.num_heads = num_heads
        self.window_size = window_size

        self.attn = InterlacedPoolAttention(
            self.dim, num_heads=num_heads, window_size=window_size, dropout=attn_drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(self.dim)
    

    def forward(self, x, 
                key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        B, P, C, H, W = x.size()
        # flatten to transformer
        x = x.permute(0, 2, 1, 3, 4).flatten(2).permute(2, 0, 1)    # [n*h*w, B, C]
        if pos != None:
            pos = pos.permute(0, 2, 1, 3, 4).flatten(2).permute(2, 0, 1)    # [n*h*w, B, C]
        if key_padding_mask != None:
            key_padding_mask = key_padding_mask.flatten(1)
     
        x = self.attn(x, key_padding_mask=key_padding_mask, pos=pos)
        
        # reshape back
        x = x.permute(0, 2, 1).contiguous().view(B, C, P, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B*P, C, H, W)
        return x


def get_default_encoder(cfg, encoder_layers_num):
    add_dim=0
    if cfg.MODEL.MULTI_POS_EMBEDDING == 'cat_vec' and cfg.MODEL.USE_MULTI_POS:
        add_dim = cfg.MODEL.MULTI_POS_EMBEDDING_DIM
    encoder_layer = TransformerEncoderLayerDefault(
                d_model=cfg.MODEL.DIM_MODEL+add_dim, nhead=cfg.MODEL.N_HEAD, dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD,
                activation='relu', normalize_before=cfg.MODEL.NORMALIZE_BEFORE)
    encoder = TransformerEncoderDefault(
            encoder_layer, encoder_layers_num)
    return encoder


def get_hrformer_encoder(cfg):
    add_dim=0
    if cfg.MODEL.MULTI_POS_EMBEDDING == 'cat_vec' and cfg.MODEL.USE_MULTI_POS:
        add_dim = cfg.MODEL.MULTI_POS_EMBEDDING_DIM
    encoder = GeneralTransformerBlock(cfg.MODEL.DIM_MODEL+add_dim, cfg.MODEL.N_HEAD, window_size=cfg.MODEL.WINDOW_SIZE)
    return encoder
    

def get_encoder(cfg, encoder_layers_num, **kwargs):
    
    if cfg.MODEL.ATTENTION_TYPE == 'default':
        logger.info('Get the default Attention architecture')
        encoder = get_default_encoder(cfg, encoder_layers_num)
    else:
        logger.info('Get the window Attention architecture')
        encoder = get_hrformer_encoder(cfg)
    
    return encoder