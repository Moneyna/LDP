import inspect
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import os

from diffusers.models import UNet2DModel, VQModel

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import PIL_INTERPOLATION, is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from LDP_arch import LDP

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def preprocess(image):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class LDMSuperResolutionPipeline_cond(DiffusionPipeline):
    r"""
    A pipeline for image super-resolution using latent diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`EulerDiscreteScheduler`],
            [`EulerAncestralDiscreteScheduler`], [`DPMSolverMultistepScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vqvae: VQModel,
        unet: UNet2DModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)
        

    def init_condition(self,model_path,dps_scale):
        self.ldp_model = LDP(in_nc=3, out_nc=3, upscele=4,
                               Nd=32,
                               d_model=64,
                               DP_depth=2,
                               diffloss_d=3,
                               diffloss_w=64,
                               diffusion_batch_mul=1,
                               num_sampling_steps='200',
                               ).to(self.device)
        self.ldp_model.load_state_dict(torch.load(model_path)['params'], strict=True)
        self.dps_scale = dps_scale


    def resize_lq(self,lq,x_up):
        H_lq, W_lq = lq.shape[2], lq.shape[3]
        H_up, W_up = x_up.shape[2], x_up.shape[3]

        if H_up > H_lq:
            x_up = x_up[:, :, :H_lq, :]
        if W_up > W_lq:
            x_up = x_up[:, :, :, :W_lq]

        if H_up < H_lq:
            pad_h = H_lq - H_up
            x_up = F.pad(x_up, (0, 0, 0, pad_h))
        if W_up < W_lq:
            pad_w = W_lq - W_up
            x_up = F.pad(x_up, (0, pad_w, 0, 0))
        return lq-x_up

    def init_image_residual(self, image):
        image = (image +1.0)/2.0

        x_down = F.interpolate(image, scale_factor=(1.0 / 2, 1.0 / 2), mode='nearest')
        x_up = F.interpolate(x_down, scale_factor=(2, 2), mode='nearest')

        self.LR_input = self.resize_lq(image, x_up)
        self.LR_input = self.LR_input * 2.0 - 1.0

    def grad_and_value(self, x_tp1,x_t, x_0_hat,lq_ori, **kwargs):

        lq_ori = (lq_ori+1.0)/2.0
        image = self.vqvae.decode(x_0_hat).sample
        image = torch.clamp(image, -1.0, 1.0)

        self.ldp_model = self.ldp_model.to(x_tp1.device)
        image = image.to(x_tp1.device)
        self.LR_input = self.LR_input.to(x_tp1.device)

        lq_pred = self.ldp_model(self.LR_input,image,True)
        lq_pred = (lq_pred +1.0) /2.0

        norm = self.dps_loss(lq_pred,lq_ori)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t)[0]
        x_tp1 -= norm_grad * self.dps_scale

        return x_tp1, norm

    def __call__(
        self,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        batch_size: Optional[int] = 1,
        num_inference_steps: Optional[int] = 100,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        duald_path = None,
        dps_loss=None,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image` or tensor representing an image batch to be used as the starting point for the process.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> import requests
        >>> from PIL import Image
        >>> from io import BytesIO
        >>> from diffusers import LDMSuperResolutionPipeline
        >>> import torch

        >>> # load model and scheduler
        >>> pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
        >>> pipeline = pipeline.to("cuda")

        >>> # let's download an  image
        >>> url = (
        ...     "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
        ... )
        >>> response = requests.get(url)
        >>> low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        >>> low_res_img = low_res_img.resize((128, 128))

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
        >>> # save image
        >>> upscaled_image.save("ldm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        #self.vqvae.train()
        #self.unet.train()

        self.dps_loss = dps_loss

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError(f"`image` has to be of type `PIL.Image.Image` or `torch.Tensor` but is {type(image)}")

        if isinstance(image, PIL.Image.Image):
            image = preprocess(image).to(self.device)
            self.init_image_residual(image)

        if duald_path != None:
            self.init_condition(duald_path)

        height, width = image.shape[-2:]

        # in_channels should be 6: 3 for latents, 3 for low resolution image
        latents_shape = (batch_size, self.unet.config.in_channels // 2, height, width)
        latents_dtype = next(self.unet.parameters()).dtype

        latents = randn_tensor(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        latents = latents.requires_grad_(True)

        # set timesteps and move to the correct device
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps_tensor = self.scheduler.timesteps

        # scale the initial noise by the standard deviation required by the scheduler
        #latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature.
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in timesteps_tensor:
            # concat latents and low resolution image in the channel dimension.
            latents = latents.requires_grad_()
            latents_input = torch.cat([latents, image], dim=1).requires_grad_(True)
            noise_pred = self.unet(latents_input, t,return_dict=False)[0] #.sample
            # compute the previous noisy sample x_t -> x_t-1
            latents_param = self.scheduler.step(noise_pred, t, latents, **extra_kwargs)
            latents1,x0_pred=latents_param[0],latents_param[1]

            latents, distance = self.grad_and_value(latents1,latents,x0_pred,image)
            latents = latents.detach_()

            if XLA_AVAILABLE:
                xm.mark_step()

        # decode the image latents with the VQVAE
        image = self.vqvae.decode(latents).sample
        image = torch.clamp(image, -1.0, 1.0)
        image = image / 2 + 0.5
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
