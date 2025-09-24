import numpy as np
import torch

from data.degradation_util import degradationBenchmarkPlus
from data.transforms import augment
from utils import img2tensor,tensor2img, imwrite

import cv2
import argparse

import glob
import os

def create_lq(gt_path,scale=4,plus=False,use_flip=False,use_rot=False,d_mode='bsrgan'):

    img_name = os.path.basename(gt_path)

    img_gt = cv2.imread(gt_path).astype(np.float32) / 255.

    img_gt = img_gt[:, :, [2, 1, 0]] # BGR to RGB

    img_lq,img_gt = degradationBenchmarkPlus(img_gt, sf=scale, lq_patchsize_h=img_gt.shape[0] // scale,lq_patchsize_w=img_gt.shape[1] // scale, use_crop=False,d_mode=d_mode)
    img_gt, img_lq = augment([img_gt, img_lq], use_flip,use_rot)
    img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

    return {
        'lq': img_lq,
        'gt': img_gt,
        'img_name':img_name
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--d_mode', type=str, default='bsrgan',help='downsample,noise,blur,JPEG')
    parser.add_argument('-s', '--scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--datasets',nargs='+',default=['DIV2K_VAL']) #'Set5','Set14','B100','Urban100'，Div2K,Div2kVal,Filkr2k #
    parser.add_argument('--use_flip', action='store_true')
    parser.add_argument('--use_rot', action='store_true')
    parser.add_argument('--mode', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output, exist_ok=True)

    for idx,dataset in enumerate(args.datasets):

        gt_base_path = os.path.join(args.input)

        gt_paths = sorted(glob.glob(os.path.join(gt_base_path, '*.png')))

        os.makedirs(os.path.join(args.output,args.d_mode,dataset,'HR'), exist_ok=True)
        os.makedirs(os.path.join(args.output, args.d_mode, dataset, 'LR','x'+str(args.scale)), exist_ok=True)

        for idx, path in enumerate(gt_paths):

            t_data=create_lq(path,scale=args.scale,plus=args.plus,use_flip=args.use_flip,use_rot=args.use_rot,d_mode = args.d_mode)
            img_name=t_data['img_name']
            (name,suffix)=os.path.splitext(img_name)

            hq_path=os.path.join(args.output, args.d_mode,dataset,'HR',f'{img_name}')
            lq_path = os.path.join(args.output, args.d_mode, dataset, 'LR','x'+str(args.scale),name+"_x"+str(args.scale)+suffix)

            hq=tensor2img(t_data['gt'])
            lq=tensor2img(t_data['lq'])

            print(lq_path)
            # save hq
            imwrite(hq, hq_path)
            # save lq
            imwrite(lq, lq_path)


if __name__ == '__main__':
    main()
