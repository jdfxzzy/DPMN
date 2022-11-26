import torch
import sys
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import string
from PIL import Image
import torchvision
from torchvision import transforms
from collections import OrderedDict
from model import tsrn, tbsrn, tatt, pgrm
from model import recognizer
from model import moran
from model import crnn
from dataset import alignCollate_realWTL, lmdbDataset_real, alignCollate_realWTLAMask
from loss import image_loss
import model.VisionLAN.cfgs.cfgs_eval as cfgs
from utils.labelmaps import get_vocabulary

sys.path.append('../')
from utils import ssim_psnr, utils_moran, utils_crnn
import dataset.dataset as dataset


class TextBase(object):

    def __init__(self, config, args, opt_TPG=None):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        self.opt_TPG = opt_TPG

        self.align_collate = alignCollate_realWTLAMask
        self.load_dataset = lmdbDataset_real

        self.align_collate_val = alignCollate_realWTL
        self.load_dataset_val = lmdbDataset_real

        self.rec_path = args.rec_path
        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = { 
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len
        self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)

        self.depths = list(eval(self.args.depths))
        self.patch_size = list(eval(self.args.patch_size))
        self.embed_dim = list(eval(self.args.embed_dim))
        self.window_size = []
        window_size_temp = list(eval(self.args.window_size))
        layer_num_pre = 0
        for _ in self.depths:
            self.window_size.append(window_size_temp[layer_num_pre:layer_num_pre+self.args.window_num])
            layer_num_pre += self.args.window_num        
        num_heads_temp = list(eval(self.args.num_heads))
        self.num_heads = []
        layer_num_pre = 0
        for layer_num in self.depths:
            self.num_heads.append(num_heads_temp[layer_num_pre:layer_num_pre+layer_num])
            layer_num_pre += layer_num
        self.mlp_ratio = list(eval(self.args.mlp_ratio))
        self.drop_rate = list(eval(self.args.drop_rate))
        self.attn_drop_rate = list(eval(self.args.attn_drop_rate))
        self.drop_path_rate = list(eval(self.args.drop_path_rate))
        

    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(self.load_dataset(root=data_dir_, voc_type=cfg.voc_type, max_len=cfg.max_len, test=False))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers), pin_memory=True,
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=True) 

        return train_dataset, train_loader

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        self.args.test_data_dir
        test_dataset = self.load_dataset_val(root=dir_, voc_type=cfg.voc_type, max_len=cfg.max_len, test=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers), pin_memory=True,
            collate_fn=self.align_collate_val(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=False)
        return test_dataset, test_loader

    def generator_init(self, iter=0, mode=True, psn=False, hidden_size=64, testing=False):
        cfg = self.config.TRAIN
        if self.args.arch == 'tsrn' and psn:
            model = tsrn.TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                              STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1])
        elif self.args.arch == 'tbsrn' and psn:
            model = tbsrn.TBSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                              STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1])
        elif self.args.arch == 'tg' and psn:
            model = tsrn.TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                              STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1])
        elif self.args.arch == 'tpgsr' and psn:
            model = tsrn.TSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                 STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1])
        elif self.args.arch == 'tatt' and psn:
            model = tatt.TSRN_TL_TRANS(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                 STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                 hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1])
        elif not psn:
            model = pgrm.PGRM(patch_size=self.patch_size, embed_dim=self.embed_dim, depths=self.depths,
                              num_heads=self.num_heads, window_size=self.window_size, mlp_ratio=self.mlp_ratio,
                              drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate,
                              drop_path_rate=self.drop_path_rate, iter=iter, mode=mode, hidden_size=hidden_size)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1])
        else:
            raise ValueError
        if self.args.arch != 'bicubic':
            model = model.to(self.device)
            if cfg.ngpu > 1:
                model = nn.DataParallel(model, device_ids=range(cfg.ngpu))
                image_crit = nn.DataParallel(image_crit, device_ids=range(cfg.ngpu))
            if self.resume is not '' and (psn or testing):
                print('loading pre-trained model from %s ' % self.resume)
                if self.config.TRAIN.ngpu == 1:
                    if os.path.isdir(self.resume):
                        if psn:
                            model.load_state_dict(
                            torch.load(
                                os.path.join(self.resume, "model_{}.pth".format(self.args.arch))
                            )['state_dict_G']
                            )
                        else:
                            model.load_state_dict(
                                torch.load(
                                    os.path.join(self.resume, "model_best_" + str(iter) + ".pth")
                                )['state_dict_G']
                                )
                    else:
                        model.load_state_dict(torch.load(self.resume)['state_dict_G'])
                else:
                    if os.path.isdir(self.resume):
                        if psn:
                            model.load_state_dict(
                                {'module.' + k: v for k, v in torch.load(
                                    os.path.join(self.resume, "model_{}.pth".format(self.args.arch))
                                )['state_dict_G'].items()}
                                )
                        else:
                            model.load_state_dict(
                                {'module.' + k: v for k, v in torch.load(
                                    os.path.join(self.resume, "model_best_" + str(iter) + ".pth")
                                )['state_dict_G'].items()}
                                )
                    else:
                        model.load_state_dict(
                        {'module.' + k: v for k, v in torch.load(self.resume)['state_dict_G'].items()})
        return {'model': model, 'crit': image_crit}

    def optimizer_init(self, model, recognizer=None, distill=None):
        cfg = self.config.TRAIN

        if not distill is None:
            if type(recognizer) == list:              
                rec_params = []
                model_params = []
                distill_params = []
                for recg in recognizer:
                    rec_params += list(recg.parameters())
                if type(model) == list:
                    for m in model:
                        model_params += list(m.parameters())
                else:
                    model_params = list(model.parameters())
                if type(distill) == list:
                    for m in distill:
                        distill_params += list(m.parameters())
                else:
                    distill_params = list(distill.parameters())
                if cfg.optimizer == "Adam":
                    optimizer = optim.Adam(model_params + rec_params + distill_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))
                elif cfg.optimizer == "AdamW":
                    optimizer = optim.AdamW(model_params + rec_params + distill_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))
            else:                
                model_params = []
                if type(model) == list:
                    for m in model:
                        model_params += list(m.parameters())
                else:
                    model_params = list(model.parameters())
                if cfg.optimizer == "Adam":
                    optimizer = optim.Adam(model_params + list(recognizer.parameters()), lr=cfg.lr, betas=(cfg.beta1, 0.999))
                elif cfg.optimizer == "AdamW":
                    optimizer = optim.AdamW(model_params + list(recognizer.parameters()), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        elif not recognizer is None:
            if type(recognizer) == list:              
                rec_params = []
                model_params = []
                for recg in recognizer:
                    rec_params += list(recg.parameters())
                if type(model) == list:
                    for m in model:
                        model_params += list(m.parameters())
                else:
                    model_params = list(model.parameters())
                if cfg.optimizer == "Adam":
                    optimizer = optim.Adam(model_params + rec_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))
                elif cfg.optimizer == "AdamW":
                    optimizer = optim.AdamW(model_params + rec_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))
            else:                
                model_params = []
                if type(model) == list:
                    for m in model:
                        model_params += list(m.parameters())
                else:
                    model_params = list(model.parameters())
                if cfg.optimizer == "Adam":
                    optimizer = optim.Adam(model_params + list(recognizer.parameters()), lr=cfg.lr, betas=(cfg.beta1, 0.999))
                elif cfg.optimizer == "AdamW":
                    optimizer = optim.AdamW(model_params + list(recognizer.parameters()), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        else:
            model_params = []
            if type(model) == list:
                for m in model:
                    model_params += list(m.parameters())
            else:
                model_params = list(model.parameters())
            if cfg.optimizer == "Adam":
                optimizer = optim.Adam(model_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))
            elif cfg.optimizer == "AdamW":
                optimizer = optim.AdamW(model_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))

        return optimizer

    def tripple_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, index):
        for i in (range(min(image_in.shape[0], self.config.TRAIN.VAL.n_vis))):
            tensor_in = image_in[i][:3, :, :]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3,:,:]
            tensor_target = image_target[i][:3,:,:]
            images = ([tensor_in, tensor_out.cpu(), tensor_target.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            out_root = os.path.join(self.vis_dir)
            if not os.path.exists(out_root):
                os.mkdir(out_root)
            out_path = os.path.join(out_root, str(index))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
            im_name = im_name.replace('/', '')
            torchvision.utils.save_image(vis_im, os.path.join(out_path, im_name), padding=0)

    def test_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, str_filt):
        visualized = 0
        for i in (range(image_in.shape[0])):
            if True:
                if (str_filt(pred_str_sr[i], 'lower') != str_filt(label_strs[i], 'lower')):
                    visualized += 1
                    tensor_in = image_in[i].cpu()
                    tensor_out = image_out[i].cpu()
                    tensor_target = image_target[i].cpu()
                    transform = transforms.Compose(
                        [transforms.ToPILImage(),
                         transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                         transforms.ToTensor()]
                    )
                    tensor_in = transform(tensor_in)
                    images = ([tensor_in, tensor_out, tensor_target])
                    vis_im = torch.stack(images)
                    vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
                    out_root = os.path.join(self.vis_dir, 'display')
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    im_name = str_filt(pred_str_lr[i], 'lower')  + '_' + str_filt(pred_str_sr[i], 'lower')  + '_' + str_filt(label_strs[i], 'lower')  + '_.png'
                    im_name = im_name.replace('/', '')
                    torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)
        return visualized

    def save_checkpoint(self, netG_list, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, recognizer=None, metric="sum"):
        ckpt_path = os.path.join(self.vis_dir, 'ckpt')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        for i in range(len(netG_list)):
            netG = netG_list[i]
            if self.config.TRAIN.ngpu == 1:
                save_dict = {
                    'state_dict_G': netG.state_dict(),
                    'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                            'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
                    'best_history_res': best_acc_dict,
                    'best_model_info': best_model_info,
                    'param_num': sum([param.nelement() for param in netG.parameters()]),
                    'converge': converge_list
                }
            else:
                save_dict = {
                    'state_dict_G': netG.module.state_dict(),
                    'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                            'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
                    'best_history_res': best_acc_dict,
                    'best_model_info': best_model_info,
                    'param_num': sum([param.nelement() for param in netG.module.parameters()]),
                    'converge': converge_list
                }
            if is_best:
                torch.save(save_dict, os.path.join(ckpt_path, 'model_best_{}_{}_{}.pth'.format(metric, epoch, i)))
            else:
                torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))

        if is_best:
            if not recognizer is None:
                if type(recognizer) == list:
                    for i in range(len(recognizer)):
                        torch.save(recognizer[i].state_dict(), os.path.join(ckpt_path, 'recognizer_best_{}_{}_{}.pth'.format(metric, epoch, i)))
                else:
                    torch.save(recognizer.state_dict(), os.path.join(ckpt_path, 'recognizer_best.pth'))
        else:
            if not recognizer is None:
                if type(recognizer) == list:
                    for i in range(len(recognizer)):
                        torch.save(recognizer[i].state_dict(), os.path.join(ckpt_path, 'recognizer_{}_{}_{}.pth'.format(metric, epoch, i)))
                else:
                    torch.save(recognizer.state_dict(), os.path.join(ckpt_path, 'recognizer.pth'))        

    def MORAN_init(self, path=None):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained if path is None else path
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        if cfg.ngpu > 1:
            MORAN = nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]
        imgs_input = nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self, path=None):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        model_path = self.config.TRAIN.VAL.crnn_pretrained if path is None else path
        print('loading pretrained crnn model from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model

    def parse_crnn_data(self, imgs_input):
        imgs_input = nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def Aster_init(self, path=None):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        model_path = path if path is not None else self.config.TRAIN.VAL.rec_pretrained
        aster.load_state_dict(torch.load(model_path)['state_dict'])
        print('load pre_trained aster model from %s' % model_path)
        aster = aster.to(self.device)
        if cfg.ngpu > 1:
            aster = nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict

    def VisionLAN_init(self, path=None):
        model_VL = cfgs.net_cfgs['VisualLAN'](**cfgs.net_cfgs['args'])
        model_path = cfgs.net_cfgs['init_state_dict'] if path is None else path
        print('load pre_trained VisionLAN model from %s' % model_path)
        model_VL = model_VL.to(self.device)
        model_VL = nn.DataParallel(model_VL)
        if cfgs.net_cfgs['init_state_dict'] != None:
            fe_state_dict_ori = torch.load(model_path)
            fe_state_dict = OrderedDict()
            for k, v in fe_state_dict_ori.items():
                if 'module' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                fe_state_dict[k] = v
            model_dict_fe = model_VL.state_dict()
            state_dict_fe = {k: v for k, v in fe_state_dict.items() if k in model_dict_fe.keys()}
            model_dict_fe.update(state_dict_fe)
            model_VL.load_state_dict(model_dict_fe)
        return model_VL        

    def parse_visionlan_data(self, imgs_input):
        imgs_input = transforms.ToPILImage()(imgs_input).convert('RGB')
        imgs_input = cv2.resize(np.array(imgs_input), (256, 64))
        imgs_input = transforms.ToTensor()(imgs_input).unsqueeze(0)
        imgs_input = imgs_input.to(self.device)
        return imgs_input

class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)