import random

import torch
import torch.nn.functional as F
from torch import nn as nn

import math

from utils.registry import ARCH_REGISTRY
from torch.utils.checkpoint import checkpoint
from archs.diffusion import create_diffusion
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False,patch_size=16):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.patch_size = patch_size
        self.net = SimpleConvAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
            patch_size=patch_size
        )

        self.train_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine",predict_xstart=True,learn_sigma = False)

    def forward(self, input, z, mask=None):
        B,C,H,W = input.shape
        patch_n = H//self.patch_size * W//self.patch_size
        t = torch.randint(self.train_diffusion.num_timesteps//2, self.train_diffusion.num_timesteps, (input.shape[0]*patch_n,), device=input.device)
        model_kwargs = dict(c=z)

        loss_dict = self.train_diffusion.training_losses_patch(self.net, target=input,x_start=input, t=t, model_kwargs=model_kwargs,patch_size=self.patch_size)

        return loss_dict['output']


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y,img_size):
        B,_,C = x.shape
        H,W = img_size
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.conv(h.permute(0, 2, 1).reshape(B,C,H,W))
        h = h.reshape(B,C,H*W).permute(0, 2, 1)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.conv = nn.Conv2d(model_channels, out_channels, kernel_size=1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c,img_size):
        B, _, C = x.shape
        H, W = img_size
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.conv(x.permute(0, 2, 1).reshape(B,C,H,W))
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        return x


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

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

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

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

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


class SimpleConvAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        img_size=64,
        patch_size=16,
        norm_layer=nn.LayerNorm,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)
        self.patch_size = patch_size

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=1,
            in_chans=in_channels,
            embed_dim=model_channels,
            norm_layer=norm_layer)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=1,
            in_chans=model_channels,
            embed_dim=out_channels,
            norm_layer=norm_layer)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        B,C,_,_ = x.shape
        x_size = (x.shape[2], x.shape[3])
        x = self.input_proj(self.patch_embed(x))
        #TODO: 调整t的大小
        t = self.time_embed(t).reshape(-1,1,C).repeat(1,self.patch_size*self.patch_size,1).reshape(B,-1,C) # B,C->，B,1,C
        c = self.cond_embed(self.patch_embed(c))

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y,x_size)
        x = self.final_layer(x, y,x_size)
        x = self.patch_unembed(x,x_size)
        return x

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class ActLayer(nn.Module):
    """activation layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """

    def __init__(self, channels, relu_type='leakyrelu'):
        super(ActLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'none':
            self.func = lambda x: x * 1.0
        elif relu_type == 'silu':
            self.func = nn.SiLU(True)
        elif relu_type == 'gelu':
            self.func = nn.GELU()
        else:
            assert 1 == 0, 'activation type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """

    def __init__(self, channels, norm_type='gn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'wn':
            self.norm = lambda x: torch.nn.utils.weight_norm(x)
        elif norm_type == 'none':
            self.norm = lambda x: x * 1.0
        else:
            assert 1 == 0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ResidualBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='silu'):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )
        if in_channel != out_channel:
            self.identity = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
                                      padding=1)
        else:
            def identity(tensor):
                return tensor

            self.identity = identity

    def forward(self, input):
        res = self.conv(input)
        out = res + self.identity(input)
        return out

def reshape2T(x, B, C):
    return x.permute(0, 2, 3, 1).reshape(B, -1, C)

# Degradation Prompt
class DP(nn.Module):
    def __init__(self, in_nc, out_nc, depth=3,rank=32, **kwargs):
        super().__init__()

        # ResBlock = ResBlock_do_fft_bench
        self.rank = rank

        self.Block_list1 = nn.ModuleList()
        self.Block_list1.append(ResidualBlock(in_nc, out_nc))
        self.Block_list1.append(ResidualBlock(out_nc, out_nc))

        self.Block_list2 = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.Block_list2.append(ResidualBlock(in_nc, rank))
                self.Block_list2.append(ResidualBlock(rank, rank))
            elif i == depth - 1:
                self.Block_list2.append(nn.Conv2d(rank, rank, 3, stride=2, padding=1))
                self.Block_list2.append(ResidualBlock(rank, rank))
                self.Block_list2.append(ResidualBlock(rank, rank))
            else:
                self.Block_list2.append(nn.Conv2d(rank, rank, 3, stride=2, padding=1))
                self.Block_list2.append(ResidualBlock(rank, rank))

        self.Block_list1 = nn.Sequential(*self.Block_list1)
        self.Block_list2 = nn.Sequential(*self.Block_list2)

        self.local_dict = nn.Embedding(rank, out_nc)

        if self.local_dict is not None:
            self.local_dict.weight.data.uniform_(-1 / out_nc, 1 / out_nc)

    def forward(self, x,img_size):
        H,W = img_size
        B, C, _, _ = x.shape

        curr_dict = self.local_dict.weight
        curr_dict = curr_dict.unsqueeze(0).repeat(B, 1, 1)

        weight = self.Block_list2(x) # B,N,rank
        weight = torch.nn.functional.interpolate(weight, size=(H,W), mode='nearest')
        weight = reshape2T(weight, B, self.rank)  # B,N,rank

        HF2 = weight @ curr_dict
        HF2 = HF2.permute(0, 2, 1).reshape(B, C, H, W)

        HF1 = self.Block_list1(HF2)

        return HF1

@ARCH_REGISTRY.register()
class LDPSR(nn.Module):
    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 upscale=4,
                 Nd=32,
                 d_model=64,
                 DP_depth=2,
                 diffloss_d=3,
                 diffloss_w=64,
                 diffusion_batch_mul=4,
                 num_sampling_steps='100',
                 patch_size=16,
                 **kwargs):
        super(LDPSR, self).__init__()


        self.upscale = upscale
        self.d_model = d_model
        self.diffusion_batch_mul = diffusion_batch_mul

        self.conv_in_HR = nn.Conv2d(in_channels=in_nc, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.conv_in_ILR = nn.Conv2d(in_channels=in_nc, out_channels=d_model, kernel_size=3, stride=1, padding=1)

        self.DP = DP(in_nc=d_model, out_nc=d_model, depth=DP_depth,rank=Nd, **kwargs)

        self.diffloss = DiffLoss(target_channels=d_model, z_channels=d_model, depth=diffloss_d, width=diffloss_w, num_sampling_steps=num_sampling_steps,patch_size=patch_size)
        self.conv_out1 = ResidualBlock(d_model,d_model)
        self.conv_out2 = nn.Conv2d(d_model, out_nc, kernel_size=3, stride=1, padding=1)
    def forward_loss(self,input,z,diffusion_batch_mul):
        B,C,H,W = input.shape
        z = z.repeat(diffusion_batch_mul, 1,1,1)
        input = input.repeat(diffusion_batch_mul, 1,1,1)

        output = self.diffloss(input = input,z = z)
        return output

    def forward(self, LR, HR,test):
        LR = self.conv_in_ILR(LR)
        HR = self.conv_in_HR(HR)

        hr_size = (HR.shape[-2],HR.shape[-1])
        z = self.DP(LR,hr_size)

        if not test:
            output = self.forward_loss(input=HR, z=z, diffusion_batch_mul=self.diffusion_batch_mul)
        else:
            output = self.forward_loss(input=HR, z=z, diffusion_batch_mul=1)

        output = F.interpolate(output, scale_factor=1.0/self.upscale, mode='nearest')

        output = self.conv_out1(output)
        output = self.conv_out2(output)
        return output


    @torch.no_grad()
    def test_tile(self, ILR,HR, tile_size=240, tile_pad=16):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = ILR.shape
        output_height = height * self.upscale
        output_width = width * self.upscale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = ILR.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = ILR[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                gt_tile = HR[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                # upscale tile
                output_tile = self.test(input_tile,gt_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.upscale
                output_end_x = input_end_x * self.upscale
                output_start_y = input_start_y * self.upscale
                output_end_y = input_end_y * self.upscale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.upscale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.upscale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.upscale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.upscale


                # put tile into output image
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[:, :,
                                                                                         output_start_y_tile:output_end_y_tile,
                                                                                         output_start_x_tile:output_end_x_tile]
        return output

    @torch.no_grad()
    def test(self, ILR,HR):

        # padding to multiple of window_size * 8
        wsz = 8 // self.upscale * 16
        wsz = min(wsz, 64)
        _, _, h_old, w_old = ILR.shape
        _, _, hr_h_old, hr_w_old = HR.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        ILR = torch.cat([ILR, torch.flip(ILR, [2])], 2)[:, :, :h_old + h_pad, :]
        ILR = torch.cat([ILR, torch.flip(ILR, [3])], 3)[:, :, :, :w_old + w_pad]

        HR = torch.cat([HR, torch.flip(HR, [2])], 2)[:, :, :hr_h_old + self.upscale*h_pad, :]
        HR = torch.cat([HR, torch.flip(HR, [3])], 3)[:, :, :, :hr_w_old + self.upscale*w_pad]
        dec = self.forward(ILR,HR,test=True)
        dec = dec[..., :h_old, :w_old]

        return dec
