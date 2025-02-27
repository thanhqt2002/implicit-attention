import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from statistics import mean
import numpy as np
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import init_weights_vit_timm, _load_weights, init_weights_vit_jax
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from utils import named_apply
import copy


 
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., layerth=0, ttl_tokens=0,s_scalar=False, attention_type="implicit"):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.ttl_tokens = ttl_tokens
        # self.layerth = layerth
        self.s_scalar = s_scalar
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # sqrt (D)
        self.scale = head_dim ** -0.5
        self.attention_type = attention_type

        if self.attention_type == "implicit":
            self.A = nn.Parameter(torch.zeros(num_heads, head_dim * 4, head_dim * 4))  # Matrix Multp
            self.B = nn.Linear(dim, dim * 4, bias=qkv_bias)
            nn.init.xavier_uniform_(self.A)
            self.fixed_point_iter = 5
        elif self.attention_type == "softmax" or self.attention_type == "lipschitz":
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            raise NotImplemented
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        

    def forward(self, x):
        B, N, C = x.shape

        if self.attention_type == "implicit":
            # B, heads, N, features
            B_U = self.B(x).reshape(B, N, self.num_heads, 4 * C // self.num_heads).permute(0, 2, 1, 3)
            X = torch.zeros((B, self.num_heads, N, 4 * C // self.num_heads), device=x.device, requires_grad=False)
            
            for _ in range(self.fixed_point_iter):
                k, q, v, prev_x = torch.chunk(torch.einsum("bhni,hij->bhnj", X, self.A) + B_U, 4, dim=-1)

                attn = q @ k.transpose(-2, -1) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                # B, heads, N, features
                x = prev_x + attn @ v

                X = torch.cat((k, q, v, x), dim=-1)
        elif self.attention_type == "softmax":
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn) 
            x = (attn @ v)
        elif self.attention_type == "lipschitz":
            # Reference: https://github.com/thanhqt2002/sim/blob/d267f3dd76f493292647d4b9bd8edd199c8f9340/train.py#L83
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            k_norm = torch.cdist(q, k, p=2) ** 2
            attn = torch.exp(-k_norm)
            # attn = attn.masked_fill(self.tril[:N, :N] == 0, 0)  # TODO: we don't need mask fill here?
            denom = 1e-6 + attn.sum(dim=-1, keepdim=True)
            attn = attn / denom / N
            attn = self.attn_drop(attn)
            w_map = v / torch.sqrt(v ** 2 + 1)
            x = attn @ w_map
        else:
            raise NotImplemented
            
    
        x = x.transpose(1, 2).reshape(B,N,C)
       
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Block(nn.Module):
 
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layerth = None,ttl_tokens=0,s_scalar=False, attention_type="implicit"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop,layerth=layerth,ttl_tokens=ttl_tokens,s_scalar=s_scalar, attention_type=attention_type)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.layerth = layerth
 
    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
 
 
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
 
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',pretrained_cfg=None,pretrained_cfg_overlay=None,s_scalar=False, attention_type="implicit", num_implicit_layers=1):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.s_scalar = s_scalar
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
       
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.num_implicit_layers = num_implicit_layers
         # replace the first num_implicit_layers with implicit attention
        if self.num_implicit_layers > -1:
            print(f"Using {self.num_implicit_layers} implicit layers")
            self.blocks = nn.Sequential(*([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, layerth = i, 
                    ttl_tokens=num_patches+self.num_tokens,s_scalar=self.s_scalar, attention_type="implicit")
                for i in range(self.num_implicit_layers)]
                +[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, layerth = i, 
                    ttl_tokens=num_patches+self.num_tokens,s_scalar=self.s_scalar, attention_type="softmax")
                for i in range(self.num_implicit_layers, depth)
                ]))
        else: # all layers are implicit
            print(f"Using all {depth} implicit layers")
            self.blocks = nn.Sequential(*[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, layerth = i, 
                    ttl_tokens=num_patches+self.num_tokens,s_scalar=self.s_scalar, attention_type="implicit")
                for i in range(depth)])
        self.norm = norm_layer(embed_dim)
 
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([f
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
 
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
        self.init_weights(weight_init)
 
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            partial(init_weights_vit_jax(mode, head_bias), head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            init_weights_vit_timm
 
    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights(m)
 
    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
 
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}
 
    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist
 
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
                
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
 
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
 



