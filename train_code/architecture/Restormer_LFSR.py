import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

from train_option import opt


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    downscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]
    kernel = torch.zeros(size = [downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                        device = input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride = downscale_factor, groups = c)

class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        downscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelUnShuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
class SpectralAngularAttention(nn.Module):
    """光谱-角度联合注意力机制（修复注意力图维度不匹配问题）"""

    def __init__(self, dim, num_heads, bias, spectral_groups, angular_groups):
        super().__init__()
        self.spectral_groups = spectral_groups
        self.angular_groups = angular_groups
        self.num_heads = num_heads
        # 改为学习每个通道组的温度参数
        self.temperature = nn.Parameter(torch.ones(1, 1, 1, 1))

        # 确保光谱和角度的通道数匹配
        group_dim = dim // max(spectral_groups, angular_groups)
        self.spectral_dim = group_dim * spectral_groups
        self.angular_dim = group_dim * angular_groups

        # 调整QKV投影的输出通道数
        self.qkv_spectral = nn.Conv2d(dim, self.spectral_dim, kernel_size=1, bias=bias)
        self.qkv_angular = nn.Conv2d(dim, self.angular_dim * 2, kernel_size=1, bias=bias)

        # 深度卷积用于捕获局部关系
        self.dwconv_spectral = nn.Conv2d(self.spectral_dim, self.spectral_dim, kernel_size=3, padding=1,
                                         groups=self.spectral_dim, bias=bias)
        self.dwconv_angular = nn.Conv2d(self.angular_dim * 2, self.angular_dim * 2, kernel_size=3, padding=1,
                                        groups=self.angular_dim * 2, bias=bias)

        self.project_out = nn.Conv2d(self.spectral_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # === 光谱维度注意力 ===
        q_spectral = self.dwconv_spectral(self.qkv_spectral(x))
        q_spectral = rearrange(q_spectral, 'b (g c) h w -> b g c (h w)', g=self.spectral_groups)
        q_spectral = torch.nn.functional.normalize(q_spectral, dim=-1)
        # 计算注意力矩阵
        attn_spectral = q_spectral @ q_spectral.transpose(-2, -1)
        # 光谱注意力图
        # 动态调整温度参数形状以匹配通道组数
        temperature = self.temperature.expand(-1, self.spectral_groups, -1, -1)
        attn_spectral = attn_spectral * temperature
        attn_spectral = attn_spectral.softmax(dim=-1)  # [B, spectral_groups, c, c]

        # === 角度维度注意力 ===
        qkv_angular = self.dwconv_angular(self.qkv_angular(x))
        q_angular, k_angular = qkv_angular.chunk(2, dim=1)
        q_angular = rearrange(q_angular, 'b (g c) h w -> b g c (h w)', g=self.angular_groups)
        k_angular = rearrange(k_angular, 'b (g c) h w -> b g c (h w)', g=self.angular_groups)

        q_angular = torch.nn.functional.normalize(q_angular, dim=-1)
        k_angular = torch.nn.functional.normalize(k_angular, dim=-1)

        # 角度注意力图
        attn_angular = q_angular @ k_angular.transpose(-2, -1)
        temperature = self.temperature.expand(-1, self.angular_groups, -1, -1)
        attn_angular = attn_angular * temperature
        attn_angular = attn_angular.softmax(dim=-1)  # [B, angular_groups, c, c]

        # === 确保两种注意力图维度匹配 ===
        if self.spectral_groups != self.angular_groups:
            # 调整角度注意力图的维度以匹配光谱注意力图
            if attn_angular.size(1) != attn_spectral.size(1):
                # 计算目标分组数
                target_groups = max(self.spectral_groups, self.angular_groups)

                # 对注意力图进行重塑和平均池化，使其分组数一致
                if attn_spectral.size(1) < target_groups:
                    # 增加光谱注意力的组数
                    factor = target_groups // attn_spectral.size(1)
                    attn_spectral = attn_spectral.unsqueeze(2).expand(-1, -1, factor, -1, -1)
                    attn_spectral = attn_spectral.reshape(b, target_groups, attn_spectral.size(3),
                                                          attn_spectral.size(4))

                if attn_angular.size(1) < target_groups:
                    # 增加角度注意力的组数
                    factor = target_groups // attn_angular.size(1)
                    attn_angular = attn_angular.unsqueeze(2).expand(-1, -1, factor, -1, -1)
                    attn_angular = attn_angular.reshape(b, target_groups, attn_angular.size(3), attn_angular.size(4))

                # 如果需要减少组数，使用平均池化
                if attn_spectral.size(1) > target_groups:
                    factor = attn_spectral.size(1) // target_groups
                    attn_spectral = attn_spectral.reshape(b, target_groups, factor, attn_spectral.size(3),
                                                          attn_spectral.size(4)).mean(dim=2)

                if attn_angular.size(1) > target_groups:
                    factor = attn_angular.size(1) // target_groups
                    attn_angular = attn_angular.reshape(b, target_groups, factor, attn_angular.size(3),
                                                        attn_angular.size(4)).mean(dim=2)

        # 确保维度完全匹配后再相乘
        assert attn_spectral.shape == attn_angular.shape, \
            f"维度不匹配: spectral {attn_spectral.shape} vs angular {attn_angular.shape}"

        # === 组合注意力 ===
        combined_attn = attn_spectral * attn_angular
        out = (combined_attn @ q_spectral)
        out = rearrange(out, 'b g c (h w) -> b (g c) h w', h=h, w=w)

        return self.project_out(out)


class SpectralGatedFFN(nn.Module):
    """光谱感知的门控前馈网络（修复分组数与通道数不匹配问题）"""

    def __init__(self, dim, ffn_expansion_factor, bias, spectral_groups):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.spectral_groups = spectral_groups

        # 确保 hidden_features*2 能被 spectral_groups 整除（核心修复）
        # 若不能整除，调整 hidden_features 使其满足条件
        if (hidden_features * 2) % spectral_groups != 0:
            hidden_features = (hidden_features * 2 // spectral_groups) * spectral_groups // 2
            # 避免 hidden_features 为 0（极端情况处理）
            if hidden_features == 0:
                hidden_features = spectral_groups // 2

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 深度卷积的分组数 now 能整除输入通道数 (hidden_features*2)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            padding=1,
            groups=spectral_groups,  # 分组数
            bias=bias
        )
        self.spectral_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_features, hidden_features // 4, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features // 4, hidden_features, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        # 光谱门控机制
        spectral_weight = self.spectral_fc(x1)
        x2 = x2 * spectral_weight

        x = F.gelu(x1) * x2
        return self.project_out(x)


class LFSR_TransformerBlock(nn.Module):
    """针对光场光谱优化的Transformer块"""

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 spectral_groups=8, angular_groups=8):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SpectralAngularAttention(
            dim, num_heads, bias,
            spectral_groups=spectral_groups,
            angular_groups=angular_groups
        )

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = SpectralGatedFFN(
            dim, ffn_expansion_factor, bias,
            spectral_groups=spectral_groups
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Restormer_LFSR(nn.Module):
    """针对光场光谱重建优化的Restormer变体"""

    def __init__(self,
                 inp_channels=3 * opt.frame,
                 out_channels=opt.num_pixels * opt.num_wavelengths,
                 dim=48,
                 num_blocks=[2, 3, 3, 4],
                 num_refinement_blocks=3,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 spectral_groups=8,
                 angular_groups=8):
        super().__init__()

        # 计算各层的光谱/角度分组数（随深度增加）
        self.spectral_groups = [
            max(spectral_groups // (2 ** i), 4)
            for i in range(len(num_blocks))
        ]
        self.angular_groups = [
            max(angular_groups // (2 ** i), 2)
            for i in range(len(num_blocks))
        ]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 编码器
        self.encoder_level1 = self._make_layer(
            dim, num_blocks[0], heads[0], ffn_expansion_factor,
            bias, LayerNorm_type, 0
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = self._make_layer(
            int(dim * 2 ** 1), num_blocks[1], heads[1], ffn_expansion_factor,
            bias, LayerNorm_type, 1
        )
        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = self._make_layer(
            int(dim * 2 ** 2), num_blocks[2], heads[2], ffn_expansion_factor,
            bias, LayerNorm_type, 2
        )
        self.down3_4 = Downsample(int(dim * 2 ** 2))

        self.latent = self._make_layer(
            int(dim * 2 ** 3), num_blocks[3], heads[3], ffn_expansion_factor,
            bias, LayerNorm_type, 3
        )

        # 解码器
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = self._make_layer(
            int(dim * 2 ** 2), num_blocks[2], heads[2], ffn_expansion_factor,
            bias, LayerNorm_type, 2
        )

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = self._make_layer(
            int(dim * 2 ** 1), num_blocks[1], heads[1], ffn_expansion_factor,
            bias, LayerNorm_type, 1
        )

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = self._make_layer(
            int(dim * 2 ** 1), num_blocks[0], heads[0], ffn_expansion_factor,
            bias, LayerNorm_type, 0
        )

        self.refinement = self._make_layer(
            int(dim * 2 ** 1), num_refinement_blocks, heads[0], ffn_expansion_factor,
            bias, LayerNorm_type, 0
        )

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def _make_layer(self, dim, num_blocks, num_heads, ffn_expansion_factor, bias, LayerNorm_type, level_idx):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(LFSR_TransformerBlock(
                dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                spectral_groups=self.spectral_groups[level_idx],
                angular_groups=self.angular_groups[level_idx]
            ))
        return nn.Sequential(*blocks)

    def forward(self, inp_img):
        # 保持与原始Restormer相同的预处理和结构
        b, c, h_inp, w_inp = inp_img.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        inp_img = F.pad(inp_img, [0, pad_w, 0, pad_h], mode='reflect')

        # 编码路径
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # 解码路径
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1[:, :, :h_inp, :w_inp]
