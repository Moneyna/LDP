# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import inspect
import os
os.environ['CUDA_VISIBLE_DEVICES']="3"
import math
from torchvision.utils import make_grid
import torch
from diffusers import LDMSuperResolutionPipeline
from pipeline_latent_diffusion_superresolution_condition import LDMSuperResolutionPipeline_cond
import argparse
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import cv2
import glob
import pyiqa
import numpy as np

from DPS_loss import DWTHFLoss

# 0.1.10
val={'metrics':{'psnr':{'type':'psnr','crop_border':0,'test_y_channel':True,'color_space':'ycbcr'},
                'ssim':{'type':'ssim','crop_border':0,'test_y_channel':True,'color_space':'ycbcr'},
                'lpips':{'type':'lpips','better':'lower'},
                'qalign':{'type':'qalign'}
}}

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')


def change_module(module_A, obj_b):
    # 获取module_A的构造函数参数签名
    init_signature = inspect.signature(module_A.__init__)
    params = {}

    # 遍历参数，跳过'self'
    for param_name, param in init_signature.parameters.items():
        if param_name == 'self':
            continue
        # 检查obj_b是否具有该属性
        if hasattr(obj_b, param_name):
            params[param_name] = getattr(obj_b, param_name)
        # 可选：处理有默认值的参数
        elif param.default != inspect.Parameter.empty:
            # 使用默认值（可选）
            params[param_name] = param.default

    # 移除不需要的参数（例如内部参数）
    params.pop('_internal_param', None)

    return params

def main(args,gt_h,gt_w):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(args.output)
    os.makedirs(args.output, exist_ok=True)

    metric_funcs = {}
    for _, opt in val['metrics'].items():
        mopt = opt.copy()
        name = mopt.pop('type', None)
        mopt.pop('better', None)
        # pyiqa 0.1.5
        metric_funcs[name] = pyiqa.create_metric(name, device='cuda', **mopt)
        # pyiqa 0.1.3
        # metric_funcs[name] = pyiqa.create_metric(name, **mopt)

    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    # load model and scheduler
    if args.mode != 'DPS':
        pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    else:
        pipeline = LDMSuperResolutionPipeline_cond.from_pretrained(model_id)


        # pipeline.vqvae = change_module(pipeline.vqvae,per_VQModel())
        # pipeline.unet = change_module( pipeline.unet,per_UNet2DModel())
        #pipeline.unet = per_UNet2DModel(**change_module(UNet2DModel, pipeline.unet))

        #for param in pipeline.unet.parameters():
        #    print(param.requires_grad)

        ldpsr_model = "/path/to/ldpsr_model.pth"
        pipeline.init_condition(ldpsr_model,args.dps_scale) #1e-2)
    sr_model = pipeline.to(device)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*.png')))
        # gt_paths = sorted(glob.glob(os.path.join(args.gt, '*.png')))

    #print("paths.len=",len(paths))
    #TODO: 读取gt
    pbar = tqdm(total=len(paths), unit='image')
    metric_results = {
        metric: 0
        for metric in val['metrics'].keys()
    }
    dps_loss = DWTHFLoss(1, 1, 1, 100)
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)

        if args.gt:
            gt_path = os.path.join(args.gt,img_name[:-7]+'.png') # xxxx_x4.png

        pbar.set_description(f'Test {img_name}')

        if args.gt:
            gt=cv2.imread(gt_path,cv2.IMREAD_UNCHANGED)
            gt_tensor = img2tensor(gt).to(device) / 255.
            gt_tensor = gt_tensor.unsqueeze(0)

            gt_tensor = F.interpolate(gt_tensor, size=(gt_h,gt_w), mode='nearest')
        # Sample images:
        img_tensor = Image.open(path).convert('RGB')
        img_tensor = img_tensor.resize((gt_h//args.out_scale,gt_w//args.out_scale), Image.Resampling.LANCZOS)

        with torch.enable_grad():
            if args.mode != 'DPS':
                output = sr_model(img_tensor,num_inference_steps=args.num_sampling_steps,eta=1).images[0]
            else:
                output = sr_model(img_tensor, num_inference_steps=args.num_sampling_steps, eta=1,dps_loss=dps_loss).images[0]
        output=cv2.cvtColor(np.array(output),cv2.COLOR_RGB2BGR)

        output=img2tensor(output).to(device) / 255.
        output = output.unsqueeze(0)

        if args.gt:

            H,W=gt_tensor.shape[2:]
            h,w=output.shape[2:]
            new_h=min(h,H)
            new_w=min(w,W)

            metric_data = [output[:,:,:new_h,:new_w].cpu(), gt_tensor[:,:,:new_h,:new_w].cpu()]
        for name, opt_ in val['metrics'].items():
            if args.gt ==None and not any(item in name for item in ['niqe','maniqa','topiq_nf','clipiqa','musiq','qalign']):
                continue
            if any(item in name for item in ['niqe','maniqa','topiq_nf','clipiqa','musiq','qalign']):
                tmp_result=metric_funcs[name](output)
            else:
                tmp_result = metric_funcs[name](*metric_data)
            metric_results[name] += tmp_result.item()

        output_img = tensor2img(output)
        if args.save_imgs:
            save_path = os.path.join(args.output, f'{img_name}')
            imwrite(output_img, save_path)
            # LR save
            #imwrite(tensor2img(img_tensor),os.path.join(args.output,img_name[:-7]+'_LR.png'))
        pbar.update(1)
    pbar.close()
    for metric in metric_results.keys():
        metric_results[metric] /= (idx + 1)
    print(metric_results)
    print(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-igt', '--gt', type=str, default=None, help='gt image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--save_imgs', action='store_true', help="save results or not")
    parser.add_argument('--mode', type=str, default='ORI',help='ORI or FT')
    parser.add_argument('--dps_scale', type=float, default=0.0)
    parser.add_argument('--strength', type=float, default=0.2)

    args = parser.parse_args()
    gt_h,gt_w = 512,512
    main(args,gt_h,gt_w)
