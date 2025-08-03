import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from modules import (
    EfficientMaskedConv1d,
    SLConv,
    get_in_mask,
    get_mid_mask,
    get_out_mask,
)


class CatConvBlock(nn.Module):

    def __init__(
        self,
        hidden_channel_full,
        slconv_kernel_size,
        num_scales,
        heads,
        use_fft_conv,
        padding_mode,
        mid_mask,
    ):
        super().__init__()
        self.block = nn.Sequential(
            SLConv(
                num_channels=hidden_channel_full,
                kernel_size=slconv_kernel_size,
                num_scales=num_scales,
                heads=heads,
                padding_mode=padding_mode,
                use_fft_conv=use_fft_conv,
            ),
            nn.BatchNorm1d(heads * hidden_channel_full),
            nn.GELU(),
            EfficientMaskedConv1d(
                in_channels=heads * hidden_channel_full,
                out_channels=hidden_channel_full,
                kernel_size=1,
                mask=mid_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(hidden_channel_full),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class CatConv(nn.Module):
    """
    Denoising network with structured long convolutions.
    Fixed number of layers used for all experiments.
    """

    def __init__(
        self,
        signal_length=100,
        signal_channel=1,
        time_dim=10,
        cond_channel=0,
        hidden_channel=20,
        in_kernel_size=17,
        out_kernel_size=17,
        slconv_kernel_size=17,
        num_scales=5,
        heads=1,
        num_blocks=3,
        num_off_diag=20,
        use_fft_conv=False,
        padding_mode="zeros",
        use_pos_emb=False,
    ):
        """
        Args:
            signal_length: Training signal length
            signal_channel: Number of signal channels
            time_dim: Diffusion time embedding dimensions
            cond_channel: Conditioning channels
            hidden_channel: Hidden channels per signal channel
            in_kernel_size: First convolution kernel size
            out_kernel_size: Last convolution kernel size
            slconv_kernel_size: Structured convolution kernel size
            num_scales: Number of scales for structured convolutions
            heads: Number of attention heads
            num_blocks: Number of network blocks
            num_off_diag: Off-diagonal interactions
            use_fft_conv: Use FFT convolution
            padding_mode: Padding mode ("zeros" or "circular")
            use_pos_emb: Use positional embeddings
        """

        super().__init__()
        self.signal_length = signal_length
        self.signal_channel = signal_channel
        self.time_dim = time_dim
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = hidden_channel * signal_channel
        cat_time_dim = 2 * time_dim if use_pos_emb else time_dim
        in_channel = signal_channel + cat_time_dim + cond_channel

        in_mask = get_in_mask(
            signal_channel, hidden_channel, cat_time_dim + cond_channel
        )
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, heads)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = nn.Sequential(
            EfficientMaskedConv1d(
                in_channels=in_channel,
                out_channels=hidden_channel_full,
                kernel_size=in_kernel_size,
                mask=in_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(hidden_channel_full),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                CatConvBlock(
                    hidden_channel_full=hidden_channel_full,
                    slconv_kernel_size=slconv_kernel_size,
                    num_scales=num_scales,
                    heads=heads,
                    use_fft_conv=use_fft_conv,
                    padding_mode=padding_mode,
                    mid_mask=mid_mask,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            in_channels=hidden_channel_full,
            out_channels=self.signal_channel,
            kernel_size=out_kernel_size,
            mask=out_mask,
            bias=True,
            padding_mode=padding_mode,
        )

    def forward(self, sig, t, cond=None):
        # sig: noisy input signal
        # t: time vector
        # cond: conditioning input
        if cond is not None:
            sig = torch.cat([sig, cond], dim=1)
            
        if self.use_pos_emb:
            pos_emb = TimestepEmbedder.timestep_embedding(
                torch.arange(self.signal_length, device=sig.device),
                self.time_dim,
            )
            pos_emb = repeat(pos_emb, "l c -> b c l", b=sig.shape[0])
            sig = torch.cat([sig, pos_emb], dim=1)

        time_emb = TimestepEmbedder.timestep_embedding(t, self.time_dim)
        time_emb = repeat(time_emb, "b t -> b t l", l=sig.shape[2])
        sig = torch.cat([sig, time_emb], dim=1)

        sig = self.conv_in(sig)
        for block in self.blocks:
            sig = block(sig)
        sig = self.conv_out(sig)
        return sig


class GeneralEmbedder(nn.Module):
    def __init__(self, cond_channel, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_channel, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, cond):
        cond = rearrange(cond, "b c l -> b l c")
        cond = self.mlp(cond)
        return rearrange(cond, "b l c -> b c l")


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: 1D Tensor of indices
        :param dim: Output dimension
        :param max_period: Controls minimum frequency
        :return: (N, D) Tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    """Modulate input with shift and scale parameters"""
    return x * (1 + scale) + shift


class AdaConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_scales = num_scales
        self.mid_mask = mid_mask

        self.conv = SLConv(
            self.kernel_size,
            channel,
            num_scales=self.num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
        )

        self.norm1 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        self.norm2 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel // 3, channel * 6, bias=True),
        )

        self.ada_ln[-1].weight.data.zero_()
        self.ada_ln[-1].bias.data.zero_()

    def forward(self, x, t_cond):
        y = x
        y = self.norm1(y)
        temp = self.ada_ln(rearrange(t_cond, "b c l -> b l c"))
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = rearrange(
            temp, "b l c -> b c l"
        ).chunk(6, dim=1)
        y = modulate(y, shift_tm, scale_tm)
        y = self.conv(y)
        y = x + gate_tm * y

        x = y
        y = self.norm2(y)
        y = modulate(y, shift_cm, scale_cm)
        y = x + gate_cm * self.mlp(y)
        return y


class AdaConv(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        cond_dim=0,
        hidden_channel=8,
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full // 3)
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full // 3)

    def forward(self, x, t, cond=None):
        x = self.conv_in(x)

        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)
        
        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])
            
        emb = t_emb + pos_emb

        for block in self.blocks:
            x = block(x, emb)
            
        x = self.conv_out(x)
        return x


class AdaConvBlock_FlLM(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.conv = SLConv(
            kernel_size,
            channel,
            num_scales=num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
        )
        self.norm1 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        self.norm2 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        
        # Embedding modulation
        self.ada_ln_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        
        # Conditioning modulation
        self.ada_ln_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        
        # Initialize weights to zero
        self.ada_ln_emb[-1].weight.data.zero_()
        self.ada_ln_emb[-1].bias.data.zero_()
        self.ada_ln_cond[-1].weight.data.zero_()
        self.ada_ln_cond[-1].bias.data.zero_()

    def forward(self, x, emb, cond):
        # Embedding modulation
        y = self.norm1(x)
        temp_emb = self.ada_ln_emb(rearrange(emb, "b c l -> b l c"))
        shift_emb, scale_emb = temp_emb.chunk(2, dim=-1)
        shift_emb = rearrange(shift_emb, "b l c -> b c l")
        scale_emb = rearrange(scale_emb, "b l c -> b c l")
        y = modulate(y, shift_emb, scale_emb)
        
        # Conditioning modulation
        temp_cond = self.ada_ln_cond(rearrange(cond, "b c l -> b l c"))
        shift_cond, scale_cond = temp_cond.chunk(2, dim=-1)
        shift_cond = rearrange(shift_cond, "b l c -> b c l")
        scale_cond = rearrange(scale_cond, "b l c -> b c l")
        y = modulate(y, shift_cond, scale_cond)
        
        # Main convolution
        y = self.conv(y)
        y = x + y

        x2 = self.norm2(y)
        x2 = self.mlp(x2)
        out = y + x2
        return out


class AdaConv_Res(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        hidden_channel=8,
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.cond_proj = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.res_proj = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock_Res(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full)
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full)

    def forward(self, x, t, cond=None, res=None):
        x = self.conv_in(x)
        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)
        
        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])
        time_emb = t_emb + pos_emb

        cond_proj = self.cond_proj(cond) if cond is not None else torch.zeros_like(x)
        res_proj = self.res_proj(res) if res is not None else torch.zeros_like(x)

        for block in self.blocks:
            x = block(x, time_emb, cond_proj, res_proj)
            
        x = self.conv_out(x)
        return x


class AdaConvBlock_Res(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.conv = SLConv(
            kernel_size,
            channel,
            num_scales=num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
        )
        self.norm1 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        self.norm2 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        
        # Combined modulation
        self.ada_ln_mix = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        
        # Residual modulation
        self.ada_ln_res = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        
        # Initialize weights to zero
        self.ada_ln_mix[-1].weight.data.zero_()
        self.ada_ln_mix[-1].bias.data.zero_()
        self.ada_ln_res[-1].weight.data.zero_()
        self.ada_ln_res[-1].bias.data.zero_()

    def forward(self, x, time_emb, cond_emb, res_emb):
        # Combined modulation
        emb_mix = time_emb + cond_emb
        y = self.norm1(x)
        temp_mix = self.ada_ln_mix(rearrange(emb_mix, "b c l -> b l c"))
        shift_mix, scale_mix = temp_mix.chunk(2, dim=-1)
        shift_mix = rearrange(shift_mix, "b l c -> b c l")
        scale_mix = rearrange(scale_mix, "b l c -> b c l")
        y = modulate(y, shift_mix, scale_mix)
        y = self.conv(y)
        y = x + y

        # Residual modulation
        y2 = self.norm2(y)
        temp_res = self.ada_ln_res(rearrange(res_emb, "b c l -> b l c"))
        shift_res, scale_res = temp_res.chunk(2, dim=-1)
        shift_res = rearrange(shift_res, "b l c -> b c l")
        scale_res = rearrange(scale_res, "b l c -> b c l")
        y2 = modulate(y2, shift_res, scale_res)
        y2 = self.mlp(y2)
        out = y + y2
        return out


class AdaConvBlock_Res_Small(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.conv = SLConv(
            kernel_size,
            channel,
            num_scales=num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.norm = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        
        # Conditional gating
        self.cond_gate = nn.Sequential(
            nn.Conv1d(channel, channel, 1),
            nn.Sigmoid()
        )
        # Initialize gate bias to negative value (tends to close)
        nn.init.constant_(self.cond_gate[0].bias, -2.0)

    def forward(self, x, time_emb, cond_emb, res_emb):
        # Main branch: residual connection
        y = self.norm(x)
        y = self.conv(y)
        y = x + y

        # Conditional branch: gated adjustment
        gate = self.cond_gate(cond_emb)
        y = y + gate * cond_emb
        
        # Residual branch: direct addition
        y = y + res_emb
        return y


class AdaConv_Res_Small(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        hidden_channel=8,
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.cond_proj = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.res_proj = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock_Res_Small(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full)
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full)

    def forward(self, x, t, cond=None, res=None):
        x = self.conv_in(x)
        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)
        
        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])
        time_emb = t_emb + pos_emb

        cond_proj = self.cond_proj(cond) if cond is not None else torch.zeros_like(x)
        res_proj = self.res_proj(res) if res is not None else torch.zeros_like(x)

        for block in self.blocks:
            x = block(x, time_emb, cond_proj, res_proj)
            
        x = self.conv_out(x)
        return x