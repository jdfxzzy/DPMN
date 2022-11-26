# --------------------------------------------------------
# Prior-Guided Refinement Modules (PGRMs)
# Modified from DW-ViT (CVPR 2022) by Zuoyan Zhao
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act_1 = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.depthwise_conv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.pointwise_conv = nn.Conv2d(hidden_features, hidden_features, 1)
        self.act_2 = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_1(x)
        x = self.drop(x)
        B, HW, _ = x.size()
        x = x.view(B, -1, int(math.sqrt(HW)), int(math.sqrt(HW)))
        x = self.depthwise_conv(x)
        x = self.act_2(x)
        x = self.pointwise_conv(x)
        x = x.view(B, HW, -1)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    if len(x.shape) == 4:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    elif len(x.shape) == 5:
        D, B, H, W, C = x.shape
        x = x.view(D, B, H // window_size, window_size, W // window_size, window_size, C) 
        windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(D, -1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SKConv(nn.Module):
    def __init__(self, dim, M, r=2, act_layer=nn.GELU):
        super(SKConv, self).__init__()
        self.dim = dim
        self.channel = dim // M  
        assert dim == self.channel * M
        self.d = self.channel // r  
        self.M = M
        self.proj = nn.Linear(dim,dim) 

        self.act_layer = act_layer()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(dim,self.d) 
        self.fc2 = nn.Linear(self.d, self.M*self.channel)
        self.softmax = nn.Softmax(dim=1)
        self.proj_head = nn.Linear(self.channel, dim)

    def forward(self, input_feats):
        bs, H, W, _ = input_feats.shape
        input_groups = input_feats.permute(0, 3, 1, 2).reshape(bs, self.M, self.channel, H, W)
        feats = self.proj(input_feats.view(bs, H*W, -1))
        feats_proj = feats.permute(0, 2, 1).reshape(bs, self.dim, H, W)
        feats = self.act_layer(feats)
        feats = feats.permute(0, 2, 1).reshape(bs, self.dim, H, W)
        feats_S = self.gap(feats)
        feats_Z = self.fc1(feats_S.squeeze())
        feats_Z = self.act_layer(feats_Z)
        attention_vectors = self.fc2(feats_Z)
        attention_vectors = attention_vectors.view(bs, self.M, self.channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(input_groups * attention_vectors, dim=1)
        feats_V = self.proj_head(feats_V.reshape(bs, self.channel, H*W).permute(0, 2, 1))
        feats_V = feats_V.permute(0, 2, 1).reshape(bs, self.dim, H, W)
        output = feats_proj + feats_V
        return output

    def flops(self, H, W):
        flops = 0
        flops += H*W * self.dim * self.dim
        flops += self.dim * self.d
        flops += self.d * self.channel*self.M
        flops += self.M * self.channel * H * W
        flops += H * W * self.channel * self.dim
        return flops


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_heads, act_layer, input_resolution,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.input_resolution = input_resolution
        self.n_group = len(self.window_size)
        self.channel = self.dim // self.n_group  
        assert self.dim == self.channel * self.n_group
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.gnum_heads = num_heads // self.n_group  
        assert num_heads == self.gnum_heads * self.n_group
        self.gchannel = self.channel // self.gnum_heads  
        assert self.channel == self.gchannel * self.gnum_heads
        self.qk_scale = qk_scale
               
        self.relative_position_index = []
        for i, window_s in enumerate(self.window_size):
            relative_position_bias_params = nn.Parameter(
                torch.zeros((2 * window_s - 1) * (2 * window_s - 1), self.gnum_heads))
            trunc_normal_(relative_position_bias_params, std=.02)
            self.register_parameter('relative_position_bias_table_{}'.format(i), relative_position_bias_params) 

            Window_size = to_2tuple(window_s)
            coords_h = torch.arange(Window_size[0])
            coords_w = torch.arange(Window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Window_size[0] - 1
            relative_coords[:, :, 1] += Window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * Window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index_{}".format(i), relative_position_index)
            self.relative_position_index.append(getattr(self, "relative_position_index_{}".format(i)))

        for i in range(len(self.window_size)):
            if min(self.input_resolution) <= self.window_size[i]:
                self.shift_size[i] = 0
                self.window_size[i] = min(self.input_resolution)
            assert 0 <= self.shift_size[i] < self.window_size[i], "shift_size must in 0-window_size"
            window_s = self.window_size[i]
            if self.shift_size[i] > 0:
                H, W = self.input_resolution
                Hp = int(np.ceil(H / window_s)) * window_s
                Wp = int(np.ceil(W / window_s)) * window_s
                img_mask = torch.zeros((1, Hp, Wp, 1))
                h_slices = (slice(0, -window_s),
                            slice(-window_s, -self.shift_size[i]),
                            slice(-self.shift_size[i], None))
                w_slices = (slice(0, -window_s),
                            slice(-window_s, -self.shift_size[i]),
                            slice(-self.shift_size[i], None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(img_mask, window_s)
                mask_windows = mask_windows.view(-1, window_s * window_s)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None
            self.register_buffer("attn_mask_{}".format(i), attn_mask)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.sknet = SKConv(dim=dim, M=self.n_group, act_layer=act_layer)

    def forward(self, x_q, x_kv):
        B, H, W, C = x_q.shape
        x_q = x_q.view(B, -1, C)
        B, HW, C = x_q.shape
        q = self.q(x_q).reshape(B, HW, 1, C).permute(2, 0, 1, 3)
        q = q.reshape(1, B, H, W, C)

        B, H, W, C = x_kv.shape
        x_kv = x_kv.view(B, -1, C)
        B, HW, C = x_kv.shape
        kv = self.kv(x_kv).reshape(B, HW, 2, C).permute(2, 0, 1, 3)
        kv = kv.reshape(2, B, H, W, C)

        q_groups = q.chunk(len(self.window_size), -1)
        kv_groups = kv.chunk(len(self.window_size), -1)    
        x_groups = []
        for i, (q_group, kv_group) in enumerate(zip(q_groups, kv_groups)):
            window_s = self.window_size[i]
            pad_l = pad_t = 0
            pad_r = (window_s - W % window_s) % window_s
            pad_b = (window_s - H % window_s) % window_s
            q_group = F.pad(q_group, (0, 0, pad_l, pad_r, pad_t, pad_b))
            kv_group = F.pad(kv_group, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, _, Hp, Wp, _ = q_group.shape

            if self.shift_size[i] > 0:
                shifted_q_group = torch.roll(q_group, shifts=(-self.shift_size[i], -self.shift_size[i]),
                                               dims=(2, 3))
                shifted_kv_group = torch.roll(kv_group, shifts=(-self.shift_size[i], -self.shift_size[i]),
                                               dims=(2, 3))
            else:
                shifted_q_group = q_group
                shifted_kv_group = kv_group

            q_windows = window_partition(shifted_q_group, window_s)
            kv_windows = window_partition(shifted_kv_group, window_s)
            q_windows = q_windows.view(1, -1, window_s * window_s, self.channel)
            kv_windows = kv_windows.view(2, -1, window_s * window_s, self.channel)
            
            _, B_, N, _ = q_windows.shape
            q = q_windows.reshape(1, B_, N, self.gnum_heads, self.gchannel).permute(0, 1, 3, 2, 4)
            kv = kv_windows.reshape(2, B_, N, self.gnum_heads, self.gchannel).permute(0, 1, 3, 2, 4)

            head_dim = q.shape[-1]
            q = q[0]
            [k, v] = [x for x in kv]
            self.scale = self.qk_scale or head_dim ** -0.5
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            window_size = to_2tuple(window_s)
            relative_position_bias = getattr(self,"relative_position_bias_table_{}".format(i))[
                self.relative_position_index[i].view(-1)].view(
                window_size[0] * window_size[1], window_size[0] * window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

            if getattr(self, "attn_mask_{}".format(i)) is not None:
                nW = getattr(self, "attn_mask_{}".format(i)).shape[0]
                attn = attn.view(B_ // nW, nW, self.gnum_heads, N, N) + getattr(self, "attn_mask_{}".format(i)).unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.gnum_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, self.channel)

            x = x.view(B_, window_s, window_s, self.channel)
            q_windows = q_windows.view(B_, window_s, window_s, self.channel)
            shifted_x_q = window_reverse(q_windows, window_s, Hp, Wp)
            shifted_x_kv = window_reverse(x, window_s, Hp, Wp)  # B H W C

            if self.shift_size[i] > 0:
                x_q = torch.roll(shifted_x_q, shifts=(self.shift_size[i], self.shift_size[i]), dims=(1, 2))
                x_kv = torch.roll(shifted_x_kv, shifts=(self.shift_size[i], self.shift_size[i]), dims=(1, 2))
            else:
                x_q = shifted_x_q
                x_kv = shifted_x_kv

            x = x.reshape(B, H, W, self.channel)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()
            x_groups.append(x)

        x = torch.cat(x_groups, -1)
        x = self.sknet(x)
        x = x.view(B, self.dim, HW).permute(0, 2, 1)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.dim * 3 * self.dim
        for i in range(len(self.window_size)):
            window_s = self.window_size[i]
            N = window_s * window_s
            nW = math.ceil(H / window_s) * math.ceil(W / window_s)
            attn_flop = self.gnum_heads * N * self.gchannel * N
            attn_v_flop = self.gnum_heads * N * N * self.gchannel
            flops += nW * (attn_flop + attn_v_flop)
        flops += self.sknet.flops(H, W)
        return flops


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size.copy()
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, shift_size=self.shift_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            act_layer=act_layer, input_resolution=self.input_resolution)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_q, x_kv):
        H, W = self.input_resolution
        B, L, C = x_q.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x_kv
        x_q_ori = x_q
        x_q = self.norm1_q(x_q)
        x_kv = self.norm1_kv(x_kv)
        x_q = x_q.view(B, H, W, C)
        x_kv = x_kv.view(B, H, W, C)

        x_kv = self.attn(x_q, x_kv)

        x_kv = shortcut + self.drop_path(x_kv)
        x_kv = x_kv + self.drop_path(self.mlp(self.norm2(x_kv)))
        return x_q_ori, x_kv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"Window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        flops += self.attn.flops(H, W)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=np.zeros(len(window_size)) if (i % 2 == 0) else np.array(window_size) // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x_q, x_kv):
        for blk in self.blocks:
            if self.use_checkpoint:
                x_q, x_kv = checkpoint.checkpoint(blk, x_q, x_kv)
            else:
                x_q, x_kv = blk(x_q, x_kv)
        if self.downsample is not None:
            x_q = self.downsample(x_q)
            x_kv = self.downsample(x_kv)
        return x_q, x_kv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2) 
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.patches_resolution[0], self.patches_resolution[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class PGRM(nn.Module):

    def __init__(self, img_size=[32, 128], patch_size=[2], in_chans=3,
                 embed_dim=[96], depths=[1], num_heads=[[6]],
                 window_size=[[2, 4, 8]], mlp_ratio=[4.], qkv_bias=True, qk_scale=None,
                 drop_rate=[0.], attn_drop_rate=[0.], drop_path_rate=[0.1], iter=0, 
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, mode=True,
                 use_checkpoint=False, hidden_size=64, **kwargs):
        super().__init__()

        if not mode:
            self.prior_fusion = nn.Conv2d(2, 3, 3, 1, 1)
        self.num_layers = depths[iter]
        self.embed_dim = embed_dim[iter]
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio[iter]
        self.window_size = window_size[iter]

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size[iter], in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size[iter], in_chans=self.embed_dim, embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate[iter])

        for i in range(iter+1):
            self.register_parameter("weight_list_{}".format(i), nn.Parameter(torch.ones(1, hidden_size, 32, 128)))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate[iter], sum(depths)*2)]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=2,
                               num_heads=num_heads[iter][i_layer],
                               window_size=window_size[iter],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate[iter], attn_drop=attn_drop_rate[iter],
                               drop_path=dpr[(sum((depths[:iter]))+i_layer)*2:(sum(depths[:iter])+i_layer+1)*2],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.conv_before_upsample = nn.Sequential(nn.Conv2d(self.embed_dim, hidden_size*patch_size[iter]*patch_size[iter], 3, 1, 1),
                                                  nn.Conv2d(hidden_size*patch_size[iter]*patch_size[iter], hidden_size*patch_size[iter]*patch_size[iter], 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.upsample = nn.PixelShuffle(patch_size[iter])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        relative_position_bias_table = set()
        for i in range(len(self.window_size)):
            relative_position_bias_table.add('relative_position_bias_table_{}'.format(i))
        return relative_position_bias_table

    def forward(self, x_q, x_kv, residual_list):
        if x_q.size(1) == 2:
            x_q = self.prior_fusion(x_q)
        x_q = self.patch_embed(x_q)
        x_kv = self.patch_embed(x_kv)
        if self.ape:
            x_q = x_q + self.absolute_pos_embed
            x_kv = x_kv + self.absolute_pos_embed
        x_q = self.pos_drop(x_q)
        x_kv = self.pos_drop(x_kv)

        for layer in self.layers:
            x_q, x_kv = layer(x_q, x_kv)
        x = self.patch_unembed(x_kv)
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = x * self.weight_list_0
        for i in range(1, len(residual_list)):
            x = x + (residual_list[i] * getattr(self, "weight_list_{}".format(i)))
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for _, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
