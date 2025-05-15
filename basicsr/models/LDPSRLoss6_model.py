from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.utils as tvu

from archs import build_network
from losses import build_loss
from utils import get_root_logger, imwrite, tensor2img, img2tensor
from utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy

import pyiqa
import numpy as np

@MODEL_REGISTRY.register()
class DualDiffLoss6Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        #self.awl = AutomaticWeightedLoss(3)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.net_g_best = build_network(opt['network_g'])
        self.net_g_best = self.model_to_device(self.net_g_best)

        self.scale = self.opt['scale']
        self.diffusion_batch_mul = self.opt['network_duald']['diffusion_batch_mul']

        # define DualDSR
        self.duald = build_network(opt['network_duald'])
        self.duald = self.model_to_device(self.duald)
        load_path = self.opt['path'].get('pretrain_network_duald', None)
        logger = get_root_logger()
        param_key = self.opt['path'].get('param_key', 'params')
        if load_path is not None:
            logger.info(f'Loading net_duald from {load_path}')
            self.load_network(self.duald, load_path, self.opt['path']['strict_load'], param_key=param_key)
        # self.duald.eval()


        # define metric functions
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items():
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                # self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)
                self.metric_funcs[name] = pyiqa.create_metric(name, **mopt)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        param_key = self.opt['path'].get('net_g_param_key', 'params')
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            ignore_keywords = self.opt['network_g'].get('ignore_keywords', None)
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'], param_key=param_key,
                              ignore_keys=ignore_keywords)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw == name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train'].get('gan_opt'))
            if self.opt['network_d'] is not None:
                self.net_d_best = copy.deepcopy(self.net_d)

        # self.net_g_best = copy.deepcopy(self.net_g)
        # self.net_g_best=build_network(opt['network_g'])

    def init_training_settings(self):
        logger = get_root_logger()

        self.ema_decay = self.opt['train'].get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            for p in self.net_g_ema.parameters():
                p.requires_grad = False
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, True,'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        if self.opt['network_d'] is not None:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            # load pretrained d models
            load_path = self.opt['path'].get('pretrain_network_d', None)
            # print(load_path)
            if load_path is not None:
                logger.info(f'Loading net_d from {load_path}')
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

            self.net_d.train()

        # define losses
        # if train_opt.get('dualdiff_opt'):
        #     self.cri_duald = build_loss(train_opt['dualdiff_opt']).to(self.device)
        # else:
        #     self.cri_duald = None

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fre_opt'):
            self.cri_fre = build_loss(train_opt['fre_opt']).to(self.device)
        else:
            self.cri_fre = None

        if train_opt.get('dwthf_opt'):
            self.cri_dwthf = build_loss(train_opt['dwthf_opt']).to(self.device)
            self.model_to_device(self.cri_dwthf)
        else:
            self.cri_dwthf = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if self.opt['network_d'] is not None:
            self.net_d_iters = train_opt.get('net_d_iters', 1)
            self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        #self.optimizer_g = optim_class([{'params':optim_params,**train_opt['optim_g']},{'params':self.awl.parameters(),'weight_decay': 0}])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        if self.opt['network_d'] is not None:
            optim_type = train_opt['optim_d'].pop('type')
            optim_class = getattr(torch.optim, optim_type)
            self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def resize_lq(self,lq,x_up):
        H_lq, W_lq = lq.shape[2], lq.shape[3]
        H_up, W_up = x_up.shape[2], x_up.shape[3]

        # 如果尺寸不同，对 x_up 进行裁剪或填充
        if H_up > H_lq:
            x_up = x_up[:, :, :H_lq, :]
        if W_up > W_lq:
            x_up = x_up[:, :, :, :W_lq]

        if H_up < H_lq:
            pad_h = H_lq - H_up
            x_up = F.pad(x_up, (0, 0, 0, pad_h))  # 只在高度方向填充
        if W_up < W_lq:
            pad_w = W_lq - W_up
            x_up = F.pad(x_up, (0, pad_w, 0, 0))  # 只在宽度方向填充
        return lq-x_up

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)

        x_down = F.interpolate(self.lq, scale_factor=(1.0 / 2, 1.0 / 2), mode='nearest')
        x_up = F.interpolate(x_down, scale_factor=(2, 2), mode='nearest')

        self.LR_input = self.resize_lq(self.lq,x_up)
        self.LR_input = self.LR_input * 2.0 - 1.0
        # self.ILR = F.interpolate(self.lq, scale_factor=(self.scale, self.scale), mode='bicubic', align_corners=False)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            # self.gt_aug = data['gt_aug'].to(self.device)

        # self.ILR = self.ILR * 2.0 - 1.0

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        if self.opt['network_d'] is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False
        self.optimizer_g.zero_grad()
        if self.ema_decay > 0:
            for p in self.net_g_ema.parameters():
                p.requires_grad = False

        sr_result = self.net_g(self.lq)

        self.result = self.duald(self.LR_input, sr_result*2.0 - 1.0)
        self.result = (self.result+1.0)/2

        self.lq = self.lq.repeat(self.diffusion_batch_mul,1,1,1)

        l_g_total = 0  # torch.zeros(1, requires_grad=True)
        loss_dict = OrderedDict()

        if self.cri_dwthf:
            l_dwthf = self.cri_dwthf(self.result, self.lq)
            l_g_total += l_dwthf
            loss_dict['l_dwthf'] = l_dwthf

        if self.cri_pix:
            l_pix = self.cri_pix(sr_result, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_fre:
            l_fre = self.cri_fre(sr_result, self.gt)
            l_g_total += l_fre
            loss_dict['l_fre'] = l_fre

        # Filter_dis
        #print("filter_dis.shape=",filter_dis.shape)
        #filter_dis = torch.sum(filter_dis)
        #l_g_total += filter_dis
        #loss_dict['l_filterD'] = filter_dis

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(sr_result, self.gt)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style

        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = self.net_d(sr_result)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)  # 不希望gan也受到加权
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan


        l_g_total.mean().backward()
        # torch.nn.utils.clip_grad_norm_(parameters=self.net_g.parameters(), max_norm=10, norm_type=2)
        self.optimizer_g.step()

        # print(">>>net_g")
        # for name, parms in self.net_g.state_dict(keep_vars=True).items():
        #    print('-->name:', name)
        #    print('-->grad_requirs:', parms.requires_grad)
        #    print('-->grad:',parms.grad)
        #    print("===")

        # optimize net_d
        if self.opt['network_d'] is not None:
            self.fixed_disc = self.opt['train'].get('fixed_disc', False)
            if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
                for p in self.net_d.parameters():
                    p.requires_grad = True
                self.optimizer_d.zero_grad()
                # real
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                loss_dict['l_d_real'] = l_d_real
                loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())

                l_d_real.backward()
                # fake
                fake_d_pred = self.net_d(sr_result.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_fake.backward()

                self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        lq_input = self.lq

        h, w = lq_input.shape[2:]
        max_size = 1024 ** 2
        if h * w < max_size:
            self.result = net_g.test(lq_input)
        else:
            self.result = net_g.test_tile(lq_input)

        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        # print("with_metrics=",with_metrics)
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            # if not hasattr(self, 'metric_results'):  # only execute in the first run
            #    self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            # self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            with torch.no_grad():
                ILR_result = self.duald.test(self.LR_input, self.result * 2.0 - 1.0)
            ILR_result = (ILR_result+1.0)/2.0

            sr_img = [tensor2img(self.result)]

            ## pyiqa==0.1.5 || 0.1.8
            metric_data = [img2tensor(sr_img[-1]).unsqueeze(0) / 255.0, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.LR_input
            del self.gt
            del self.result
            # del self.output_ema
            torch.cuda.empty_cache()

            if save_img:
                for i in range(len(sr_img)):
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                                 f'{current_iter}',
                                                 f'{img_name}_{i}.png')
                        save_post_sr_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                                 f'{current_iter}',
                                                 f'{img_name}_{i}_ILR.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(
                                self.opt['path']['visualization'], dataset_name,
                                f'{img_name}_{self.opt["val"]["suffix"]}_{i}.png')
                            save_post_sr_path = osp.join(
                                self.opt['path']['visualization'], dataset_name,
                                f'{img_name}_{self.opt["val"]["suffix"]}_{i}_ILR.png')
                        else:
                            save_img_path = osp.join(
                                self.opt['path']['visualization'], dataset_name,
                                f'{img_name}_{self.opt["name"]}_{i}.png')
                            save_post_sr_path = osp.join(
                                self.opt['path']['visualization'], dataset_name,
                                f'{img_name}_{self.opt["name"]}_{i}_ILR.png')
                    if save_as_dir:
                        save_as_img_path = osp.join(save_as_dir, f'{img_name}_{i}.png')
                        imwrite(sr_img, save_as_img_path)
                    imwrite(sr_img[i], save_img_path)
                    imwrite(tensor2img(ILR_result), save_post_sr_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if any(item in name for item in ['niqe', 'maniqa', 'topiq_nf', 'clipiqa', 'musiq', 'qalign']):
                        tmp_result = self.metric_funcs[name](self.result)
                    else:
                        tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    if self.opt['network_d'] is not None:
                        self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    if self.opt['network_d'] is not None:
                        self.save_network(self.net_d, 'net_d_best', '')
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net_g, self.net_g_best)
                    if self.opt['network_d'] is not None:
                        self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    if self.opt['network_d'] is not None:
                        self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx)
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.result.detach().cpu()[:vis_samples]

        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.opt['network_d'] is not None:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
