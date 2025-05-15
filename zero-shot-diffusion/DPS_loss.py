import numpy as np
import pywt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyiqa

def l1_loss(pred, target,reduction):
    #print("pred.shape",pred.shape)
    #print("target.shape",target.shape)
    return F.l1_loss(pred, target, reduction=reduction)

class LPIPSLoss(nn.Module):
    """LPIPS loss with vgg backbone.
    """
    def __init__(self, loss_weight = 1.0):
        super(LPIPSLoss, self).__init__()
        self.model = pyiqa.create_metric('lpips-vgg', as_loss=True)
        self.loss_weight = loss_weight

    def forward(self, x, gt):
        return self.model(x, gt) * self.loss_weight

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, reduction=self.reduction)

class FrequencyLoss(nn.Module):
    """Frequency loss.

    Modified from:
    `<https://github.com/Jiahao000/MFM/blob/master/models/frequency_loss.py>`_.

    Args:
        loss_gamma (float): the exponent to control the sharpness of the frequency distance. Defaults to 1.
        matrix_gamma (float): the scaling factor of the spectrum weight matrix for flexibility. Defaults to 1.
        patch_factor (int): the factor to crop image patches for patch-based frequency loss. Defaults to 1.
        ave_spectrum (bool): whether to use minibatch average spectrum. Defaults to False.
        with_matrix (bool): whether to use the spectrum weight matrix. Defaults to False.
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Defaults to False.
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Defaults to False.
    """

    def __init__(self,
                 loss_weight=1.0,
                 loss_gamma=1.,
                 matrix_gamma=1.,
                 patch_factor=1,
                 ave_spectrum=False,
                 with_matrix=False,
                 log_matrix=False,
                 batch_matrix=False):
        super(FrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_gamma = loss_gamma
        self.matrix_gamma = matrix_gamma
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.with_matrix = with_matrix
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1).float()  # NxPxCxHxW

        # perform 2D FFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm='ortho')
        # shift low frequency to the center
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        # stack the real and imaginary parts along the last dimension
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        loss = torch.sqrt(tmp[..., 0] + tmp[..., 1] + 1e-12) ** self.loss_gamma
        if self.with_matrix:
            # spectrum weight matrix
            if matrix is not None:
                # if the matrix is predefined
                weight_matrix = matrix.detach()
            else:
                # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
                matrix_tmp = (recon_freq - real_freq) ** 2
                matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.matrix_gamma

                # whether to adjust the spectrum weight matrix by logarithm
                if self.log_matrix:
                    matrix_tmp = torch.log(matrix_tmp + 1.0)

                # whether to calculate the spectrum weight matrix using batch-based statistics
                if self.batch_matrix:
                    matrix_tmp = matrix_tmp / matrix_tmp.max()
                else:
                    matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

                matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
                matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
                weight_matrix = matrix_tmp.clone().detach()

            assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                'The values of spectrum weight matrix should be in the range [0, 1], '
                'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))
            # dynamic spectrum weighting (Hadamard product)
            loss = weight_matrix * loss
        return loss

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate frequency loss.

        Args:
            pred (torch.Tensor): Predicted tensor with shape (N, C, H, W).
            target (torch.Tensor): Target tensor with shape (N, C, H, W).
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Defaults to None.
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate frequency loss
        loss= self.loss_formulation(pred_freq, target_freq, matrix).mean()
        return self.loss_weight * loss

def dwt_get_highfrequency2(image, wavelet_name='haar'):
    if len(image.shape) == 2:
        coeffs = pywt.dwt2(image, wavelet_name)
        LL, (cH, cV, cD) = coeffs
    elif len(image.shape) == 3:
        LL = []
        cH_list = []
        cV_list = []
        cD_list = []
        for i in range(image.shape[2]):
            coeffs = pywt.dwt2(image[:, :, i], wavelet_name)
            LL_i, (cH_i, cV_i, cD_i) = coeffs
            LL.append(LL_i)
            cH_list.append(cH_i)
            cV_list.append(cV_i)
            cD_list.append(cD_i)
        LL = np.stack(LL, axis=2)
        cH = np.stack(cH_list, axis=2)
        cV = np.stack(cV_list, axis=2)
        cD = np.stack(cD_list, axis=2)


    high_frequency = np.sqrt(cH ** 2 + cV ** 2 + cD ** 2)

    return LL, cH, cV, cD, high_frequency

def dwt_get_highfrequency(image_tensor, wavelet_name='sym2'):
    image_np = image_tensor.detach().cpu().numpy()

    batch_size = image_np.shape[0]
    high_frequency_batch = []
    for i in range(batch_size):
        image = np.transpose(image_np[i], (1, 2, 0))
        LL, cH, cV, cD, high_frequency = dwt_get_highfrequency2(image, wavelet_name)
        high_frequency = np.transpose(high_frequency, (2, 0, 1))
        high_frequency_batch.append(high_frequency)
    high_frequency_batch = np.stack(high_frequency_batch)
    return high_frequency_batch

def normalize_highfrequency(high_frequency):
    max_val = np.max(high_frequency)
    min_val = np.min(high_frequency)
    epsilon = 1e-7
    if max_val - min_val < epsilon:
        weight_map = np.zeros_like(high_frequency)
    else:
        weight_map = (high_frequency - min_val) / (max_val - min_val)
    return weight_map

class DWTHFLoss(nn.Module):
    def __init__(self, loss_weight_1=0.0,loss_weight_2=0.0,loss_weight_3=0.0,map_weight=1,reduction = 'mean',diff=False,wavelet_name='db2'):
        super().__init__()
        self.l1_loss = L1Loss(loss_weight=loss_weight_1,reduction=reduction)
        self.lpips_loss = LPIPSLoss(loss_weight=loss_weight_2)
        self.fre_loss = FrequencyLoss(loss_weight=loss_weight_3)

        self.diff = diff
        self.wavelet_name = wavelet_name
        self.map_weight = map_weight

        self.loss_weight_l1 = loss_weight_1
        self.loss_weight_lpips = loss_weight_2
        self.loss_weight_fre = loss_weight_3


    def forward(self, pred, target):
        if self.diff:
            pred = (pred+1.0)/2.0
            target = (target+1.0)/2.0

        B,C,H,W =pred.shape


        high_frequency = dwt_get_highfrequency(target,self.wavelet_name)
        weight_map = normalize_highfrequency(high_frequency)
        weight_map = torch.from_numpy(weight_map).to(target.device)
        weight_map = F.interpolate(weight_map, scale_factor=(2, 2), mode='nearest')
        weight_map = weight_map[:,:,:H,:W]

        weighted_I_hat_test_LR = self.map_weight * weight_map * target
        weighted_I_test_LR = self.map_weight * weight_map * pred

        if self.loss_weight_l1 != 0.0:
            l1_loss = self.l1_loss(weighted_I_hat_test_LR, weighted_I_test_LR)
        else:
	    l1_loss = 0.0

        if self.loss_weight_lpips != 0.0:
            lpips_loss = self.lpips_loss(weighted_I_hat_test_LR, weighted_I_test_LR)
        else:
            lpips_loss = 0.0

        if self.loss_weight_fre != 0.0:
            fre_loss = self.fre_loss(weighted_I_hat_test_LR, weighted_I_test_LR)
        else:
            fre_loss = 0.0

        total_loss = l1_loss + lpips_loss+fre_loss

        return total_loss