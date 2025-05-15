import os
import cv2
from PIL import Image
import random
import numpy as np
from torch.utils import data as data

from data.transforms import augment, paired_random_crop
from utils import FileClient, img2tensor
from utils.registry import DATASET_REGISTRY

from .data_util import make_dataset,input_mask_with_noise
import torch.nn.functional as F


def random_resize(img, scale_factor_w=1.,scale_factor_h=1.):
    return cv2.resize(img, None, fx=scale_factor_w, fy=scale_factor_h, interpolation=cv2.INTER_CUBIC)

def paired_resize(img_gt,img_lr, gt_size, scale_factor):
    new_gt=cv2.resize(img_gt, (gt_size,gt_size), interpolation=cv2.INTER_CUBIC)
    lr_size=gt_size//scale_factor
    new_lr=cv2.resize(img_lr, (lr_size,lr_size), interpolation=cv2.INTER_CUBIC)
    return new_gt,new_lr

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
                
        self.lq_paths = make_dataset(self.lq_folder)
        self.gt_paths = make_dataset(self.gt_folder)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        #image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        # open-cv
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.

        lq_path = self.lq_paths[index]
        if gt_path != lq_path:
            t1=gt_path.split("/")
            t2=lq_path.split("/")
            t2[-1]=t1[-1][:-4]+t2[-1][-7:]
            lq_path='/'.join(t2)
            #lq_path=lq_path.replace(t2[-1][:-7],t1[-1][:-4])
            #print("lq_path=",lq_path)
            #print("gt_path=",gt_path)
        # opencv
        img_lq = cv2.imread(lq_path).astype(np.float32) / 255.

        # augmentation for training
        if self.opt['phase'] == 'train':
            #input_gt_size = img_gt.shape[0]
            #input_lq_size = img_lq.shape[0]
            input_gt_h,input_gt_w = img_gt.shape[0],img_gt.shape[1]
            input_lq_h,input_lq_w = img_lq.shape[0],img_lq.shape[1]
            gt_size = self.opt['gt_size']

            if input_gt_h < gt_size or input_gt_w < gt_size:
                pad_h = max(0, gt_size - input_gt_h)
                pad_w = max(0, gt_size - input_gt_w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h//scale, 0, pad_w//scale, cv2.BORDER_REFLECT_101)

            if self.opt['use_resize_crop']:
                # random resize
                #input_gt_random_size = random.randint(gt_size, input_gt_size)
                #input_gt_random_size = input_gt_random_size - input_gt_random_size % scale # make sure divisible by scale 
                #resize_factor = input_gt_random_size / input_gt_size
                #img_gt = random_resize(img_gt, resize_factor)
                #img_lq = random_resize(img_lq, resize_factor)
                input_gt_h, input_gt_w = img_gt.shape[0], img_gt.shape[1]
                input_lq_h, input_lq_w = img_lq.shape[0], img_lq.shape[1]
                # h
                input_gt_random_h = random.randint(gt_size, input_gt_h)
                input_gt_random_h = input_gt_random_h - input_gt_random_h % scale # make sure divisible by scale
                resize_factor_h = input_gt_random_h / input_gt_h
                # w
                input_gt_random_w = random.randint(gt_size, input_gt_w)
                input_gt_random_w = input_gt_random_w - input_gt_random_w % scale  # make sure divisible by scale
                resize_factor_w = input_gt_random_w / input_gt_w
                # random_resize
                img_gt = random_resize(img_gt, scale_factor_w=resize_factor_w,scale_factor_h=resize_factor_h)
                img_lq = random_resize(img_lq, scale_factor_w=resize_factor_w,scale_factor_h=resize_factor_h)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

                #TODO:Mask
                if self.opt.get('if_mask'):
                    img_lq = input_mask_with_noise(img_lq, mask1=self.opt['mask1'], mask2=self.opt['mask2'])
                    #print("mask_done[1]!")
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        if self.opt['phase'] != 'train':
            crop_eval_size = self.opt.get('crop_eval_size', None)
            if crop_eval_size:
                img_gt, img_lq = paired_resize(img_gt, img_lq, crop_eval_size, scale)

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        upsample = self.opt.get('upsample', None)
        if upsample:
            img_lq = F.interpolate(img_lq.unsqueeze(0), scale_factor=(scale, scale), mode='bicubic',
                                   align_corners=False)
            # img_lq = img_lq

        temp={
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }
 
        #print("datas=",temp)
        return temp

    def __len__(self):
        return len(self.gt_paths)
