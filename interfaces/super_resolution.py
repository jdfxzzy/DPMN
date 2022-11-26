import torch
import sys
import time
import os
import csv
import math
from datetime import datetime
import copy
import numpy as np
import pygame
from pygame import freetype
sys.path.append('../')
sys.path.append('./')
from interfaces import base
from utils.metrics import get_str_list
from utils.util import str_filt, toMask, torch_rotate_img
from utils.render_standard_text import make_standard_text
from model.VisionLAN.utils import Attention_AR_counter
from model.cmm import ComplementationModulationModule
from model.distill_module import DistillModule
import model.VisionLAN.cfgs.cfgs_eval as cfgs


class TextSR(base.TextBase):
    def train(self):
        TP_Generator_dict = {
            "crnn": self.CRNN_init,
            "aster": self.Aster_init,
            "moran": self.MORAN_init,
            "visionlan": self.VisionLAN_init
        }
        pygame.init()
        freetype.init()

        cfg = self.config.TRAIN
        _, train_loader = self.get_train_data()
        _, val_loader_list = self.get_val_data()
        model_dict = self.generator_init(0, mode=False, hidden_size=3)
        model, image_crit = model_dict['model'], model_dict['crit']

        stu_iter_b1 = self.args.stu_iter_b1
        stu_iter_b2 = self.args.stu_iter_b2

        model_list = [model]
        if not self.args.sr_share:
            for i in range(stu_iter_b1-1):
                model_sep = self.generator_init(i+1, mode=False, hidden_size=3)['model']
                model_list.append(model_sep)

        model_list.append(self.generator_init(stu_iter_b1, mode=True, hidden_size=3)['model'])
        if not self.args.sr_share:
            for i in range(stu_iter_b1, stu_iter_b1+stu_iter_b2-1):
                model_sep = self.generator_init(i+1, mode=True, hidden_size=3)['model']
                model_list.append(model_sep)

        model_psn = self.generator_init(0, psn=True)['model']
        for p in model_psn.parameters():
            p.requires_grad = False
        model_psn.eval()
        
        aster_info = None
        if self.args.rec == 'moran':
            rec = self.MORAN_init()
        elif self.args.rec == 'aster':
            rec, aster_info = self.Aster_init()
        elif self.args.rec == 'crnn':
            rec = self.CRNN_init()
        for p in rec.parameters():
            p.requires_grad = False
        rec.eval()
        
        model_cmm = ComplementationModulationModule()
        model_cmm = model_cmm.to(self.device)
        if self.config.TRAIN.ngpu > 1:
            model_cmm = torch.nn.DataParallel(model_cmm, device_ids=range(self.config.TRAIN.ngpu))
        model_list.append(model_cmm)

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
            
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))        
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []
        log_path = os.path.join(cfg.ckpt_dir, "log.csv")

        display = True
        crnn_psn = None
        if self.args.arch == "tpgsr" or self.args.arch == "tatt":
            crnn_psn = self.CRNN_init(os.path.join(self.resume, "recognizer_best_crnn.pth"))
            crnn_psn.eval()
            for p in crnn_psn.parameters():
                p.requires_grad = False
        test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])

        recognizer_student = []
        for i in range(stu_iter_b1):
            recognizer_path = os.path.join(self.rec_path, "recognizer_best_" + str(i))
            if self.args.tpg=="aster":
                recognizer_path += ".pth.tar"
            else:
                recognizer_path += ".pth"
            recognizer_student_ = TP_Generator_dict[self.args.tpg](path=recognizer_path) 
            if type(recognizer_student_) == list:
                recognizer_student_ = recognizer_student_[i]

            recognizer_student.append(recognizer_student_)
        
        distill_list = []
        for i in range(stu_iter_b1+stu_iter_b2-2):
            distill_module = DistillModule()
            distill_module = distill_module.to(self.device)
            if self.config.TRAIN.ngpu > 1:
                distill_module = torch.nn.DataParallel(distill_module, device_ids=range(self.config.TRAIN.ngpu))
            for p in distill_module.parameters():
                p.requires_grad = True
            distill_list.append(distill_module)

        optimizer_G = self.optimizer_init(model_list, recognizer_student, distill_list)
        print('='*110)
        for epoch in range(cfg.epochs):
            for j, data in (enumerate(train_loader)):
                if display:
                    start = time.time()
                    display = False

                for model in model_list:
                    model.train()
                    for p in model.parameters():
                        p.requires_grad = True
                for model in recognizer_student:
                    model.train()
                    for p in model.parameters():
                        p.requires_grad = True

                iters = len(train_loader) * epoch + j + 1
                optimizer_G.zero_grad()
                images_hr, _, images_lr, _, _, _, _, _, _ = data
                loss_im_avg = 0
                if self.args.rotate_train:
                    batch_size = images_lr.shape[0]
                    angle_batch = np.random.rand(batch_size) * self.args.rotate_train * 2 - self.args.rotate_train
                    arc = angle_batch / 180. * math.pi
                    rand_offs = torch.tensor(np.random.rand(batch_size)).float()
                    arc = torch.tensor(arc).float()
                    images_lr = torch_rotate_img(images_lr, arc, rand_offs)
                    images_hr = torch_rotate_img(images_hr, arc, rand_offs)

                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                if self.args.arch in ['tsrn', 'tbsrn', 'tg']:
                    images_lr_psn = model_psn(images_lr)
                elif self.args.arch == 'tpgsr':
                    crnn_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                    label_vecs_logits = crnn_psn(crnn_dict_lr)
                    label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                    label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                    images_lr_psn = model_psn(images_lr, label_vecs_final * 1.)
                elif self.args.arch == 'tatt':
                    crnn_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                    label_vecs_logits = crnn_psn(crnn_dict_lr)
                    label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                    label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                    images_lr_psn, _ = model_psn(images_lr, label_vecs_final.detach())

                cascade_images = images_lr_psn
                cascade_images_list_branch_1 = []

                for k in range(stu_iter_b1):
                    x_q = torch.empty(1, 2, cascade_images.size(2), cascade_images.size(3))
                    prob_lr = torch.empty(1, 25, 37)
                    for i in range(cascade_images.size(0)):
                        visionlan_dict_lr = self.parse_visionlan_data(cascade_images[i, :3, :, :])
                        target = ''
                        label_lr, label_length = recognizer_student[k](visionlan_dict_lr, target, '', False)
                        pred_str_lr, pred_prob = test_acc_counter.convert(label_lr, label_length)
                        s = pred_str_lr[0]
                        prob_lr = torch.cat([prob_lr, pred_prob], dim=0)
                        if s == "" or type(s) == torch.Tensor:
                            s = "\t"
                        lower_case = s.lower()
                        upper_case = s.upper()
                        i_t_lower = make_standard_text(self.args.font_path, lower_case, (cascade_images.size(2), cascade_images.size(3)))
                        i_t_lower_tensor = torch.from_numpy(i_t_lower).unsqueeze(0).unsqueeze(0)
                        i_t_upper = make_standard_text(self.args.font_path, upper_case, (cascade_images.size(2), cascade_images.size(3)))
                        i_t_upper_tensor = torch.from_numpy(i_t_upper).unsqueeze(0).unsqueeze(0)
                        i_t_tensor = torch.cat([i_t_lower_tensor, i_t_upper_tensor], dim=1)
                        x_q = torch.cat([x_q, i_t_tensor], dim=0)
                      
                    x_q = x_q[1:]
                    x_kv = cascade_images[:, :3, :]
                    prob_lr = prob_lr[1:]
                    prob_lr = prob_lr.to(self.device)
                    x_q = x_q.to(self.device)
                    x_kv = x_kv.to(self.device)

                    if self.args.sr_share:
                        pick = 0
                    else:
                        pick = k

                    image_sr = model_list[pick](x_q, x_kv, cascade_images_list_branch_1[:k])
                    image_sr = image_sr.to(self.device)
                    cascade_images_list_branch_1.append(image_sr)
                    cascade_images = image_sr

                    loss_im = image_crit(image_sr, images_hr[:, :3, :]).mean() * 100
                    loss_im_avg += loss_im

                image_sr_branch1 = cascade_images_list_branch_1[-1]
                cascade_images = images_lr_psn
                cascade_images_list_branch_2 = []
                for k in range(stu_iter_b1, stu_iter_b1+stu_iter_b2):
                    x_q = torch.empty(1, 3, cascade_images.size(2), cascade_images.size(3))
                    for i in range(cascade_images.size(0)):
                        mask = toMask(cascade_images[i][:3, :])
                        x_q = torch.cat([x_q, mask], dim=0)

                    x_q = x_q[1:]
                    x_kv = cascade_images[:, :3, :]
                    x_q = x_q.to(self.device)
                    x_kv = x_kv.to(self.device)

                    if self.args.sr_share:
                        pick = 0
                    else:
                        pick = k

                    image_sr = model_list[pick](x_q, x_kv, cascade_images_list_branch_2[:(k-self.args.stu_iter_b2)])
                    image_sr = image_sr.to(self.device)
                    cascade_images_list_branch_2.append(image_sr)
                    cascade_images = image_sr

                    loss_im = image_crit(image_sr, images_hr[:, :3, :]).mean() * 100
                    loss_im_avg += loss_im

                image_sr_branch2 = cascade_images_list_branch_2[-1]
                cascade_images = image_sr_branch1

                distill_feature = cascade_images_list_branch_1[-1]
                for k in range(stu_iter_b1-1, 0, -1):
                    distill_module = distill_list[k-1]
                    x_deep = distill_feature
                    x_shallow = cascade_images_list_branch_1[k-1]
                    loss_distill, feature = distill_module(x_deep, x_shallow)
                    loss_distill = loss_distill.sum()
                    distill_feature = feature
                    loss_im_avg += loss_distill * 100

                distill_feature = cascade_images_list_branch_2[-1]
                for k in range(stu_iter_b2-1, 0, -1):
                    distill_module = distill_list[k+stu_iter_b1-2]
                    x_deep = distill_feature
                    x_shallow = cascade_images_list_branch_2[k-1]
                    loss_distill, feature = distill_module(x_deep, x_shallow)
                    loss_distill = loss_distill.sum()
                    distill_feature = feature
                    loss_im_avg += loss_distill * 100
                                        
                image_sr = model_list[-1](image_sr_branch1, image_sr_branch2)
                image_sr = image_sr.to(self.device)                       
                loss_im = image_crit(image_sr, images_hr[:, :3, :]).mean() * 100
                loss_im_avg += loss_im
                loss_im_avg /= (stu_iter_b1 + stu_iter_b2 + 1)
                loss_im_avg.backward()

                for model in model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                for model in recognizer_student:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                for model in distill_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer_G.step()

                if iters % cfg.displayInterval == 0:
                    end = time.time()
                    duration = end - start
                    display = True
                    print('[{}] | '
                          'Epoch: [{}][{} / {}] | '
                          'Loss: {} | Duration: {}s'
                            .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  float(loss_im_avg.data), duration))
                    print('-'*110)

                if iters % cfg.VAL.valInterval == 0:
                    print('='*110)
                    current_acc_dict = {}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        print('evaling %s' % data_name)
                        metrics_dict = self.eval(model_list, val_loader, epoch, rec, aster_info, recognizer_student, model_psn, crnn_psn)
                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        
                        if acc > best_history_acc[data_name]:
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            best_model_info = {'accuracy': float(acc), 'psnr': metrics_dict['psnr_avg'], 'ssim': metrics_dict['ssim_avg']}
                            print('best_%s = %.2f%% (A New Record)' % (data_name, best_history_acc[data_name] * 100))
                            self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, True, converge_list, recognizer_student, data_name)
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name, metrics_dict['accuracy'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg'], "best_{}".format(data_name)])
                        else:
                            print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow([epoch, data_name, metrics_dict['accuracy'], metrics_dict['psnr_avg'], metrics_dict['ssim_avg']])
                        print('-'*110)
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model')
                        self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, True, converge_list, recognizer_student)
                        with open(log_path, "a+", newline="") as out:
                            writer = csv.writer(out)
                            writer.writerow([epoch, "", "", "", "", "", "best_sum"])
                    print('='*110)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, False, converge_list, None)


    def eval(self, model_list, val_loader, index, rec, aster_info=None, rec_list=None, model_psn=None, crnn_psn=None):
        for model in model_list:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        for r in rec_list:
            r.eval()
            for p in r.parameters():
                p.requires_grad = False

        test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                           cfgs.dataset_cfgs['case_sensitive'])
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}

        for _, data in (enumerate(val_loader)):
            images_hr, images_lr, _, _, label_strs, _ = data
            if self.args.rotate_test:
                batch_size = images_lr.shape[0]
                angle_batch = np.random.rand(batch_size) * self.args.rotate_train * 2 - self.args.rotate_train
                arc = angle_batch / 180. * math.pi
                rand_offs = torch.tensor(np.random.rand(batch_size)).float()
                arc = torch.tensor(arc).float()
                images_lr = torch_rotate_img(images_lr, arc, rand_offs)
                images_hr = torch_rotate_img(images_hr, arc, rand_offs)

            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)

            if self.args.arch in ['tsrn', 'tbsrn', 'tg']:
                images_lr_psn = model_psn(images_lr)
            elif self.args.arch == 'tpgsr':
                crnn_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                label_vecs_logits = crnn_psn(crnn_dict_lr)
                label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                images_lr_psn = model_psn(images_lr, label_vecs_final * 1.)
            elif self.args.arch == 'tatt':
                crnn_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                label_vecs_logits = crnn_psn(crnn_dict_lr)
                label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                images_lr_psn, _ = model_psn(images_lr, label_vecs_final.detach())

            cascade_images = images_lr_psn
            cascade_images_list = []
            val_batch_size = images_lr.shape[0]

            for k in range(self.args.stu_iter_b1):
                x_q = torch.empty(1, 2, cascade_images.size(2), cascade_images.size(3))
                for i in range(cascade_images.size(0)):
                    visionlan_dict_lr = self.parse_visionlan_data(cascade_images[i, :3, :, :])
                    target = ''
                    label_lr, label_length = rec_list[k](visionlan_dict_lr, target, '', False)
                    pred_str_lr, _ = test_acc_counter.convert(label_lr, label_length)
                    s = pred_str_lr[0]
                    if s == "" or type(s) == torch.Tensor:
                        s = "\t"
                    lower_case = s.lower()
                    upper_case = s.upper()
                    i_t_lower = make_standard_text(self.args.font_path, lower_case, (cascade_images.size(2), cascade_images.size(3)))
                    i_t_lower_tensor = torch.from_numpy(i_t_lower).unsqueeze(0).unsqueeze(0)
                    i_t_upper = make_standard_text(self.args.font_path, upper_case, (cascade_images.size(2), cascade_images.size(3)))
                    i_t_upper_tensor = torch.from_numpy(i_t_upper).unsqueeze(0).unsqueeze(0)
                    i_t_tensor = torch.cat([i_t_lower_tensor, i_t_upper_tensor], dim=1)
                    x_q = torch.cat([x_q, i_t_tensor], dim=0)
                      
                x_q = x_q[1:]
                x_kv = cascade_images[:, :3, :]
                x_q = x_q.to(self.device)
                x_kv = x_kv.to(self.device)

                if self.args.sr_share:
                    pick = 0
                else:
                    pick = k

                image_sr = model_list[pick](x_q, x_kv, cascade_images_list[:k])
                image_sr = image_sr.to(self.device)
                cascade_images_list.append(image_sr)
                cascade_images = image_sr

            image_sr_branch1 = cascade_images_list[-1]
            cascade_images = images_lr_psn
            cascade_images_list = []
            for k in range(self.args.stu_iter_b1, self.args.stu_iter_b1+self.args.stu_iter_b2):
                x_q = torch.empty(1, 3, cascade_images.size(2), cascade_images.size(3))
                for i in range(cascade_images.size(0)):
                    mask = toMask(cascade_images[i][:3, :])
                    x_q = torch.cat([x_q, mask], dim=0)

                x_q = x_q[1:]
                x_kv = cascade_images[:, :3, :]
                x_q = x_q.to(self.device)
                x_kv = x_kv.to(self.device)

                if self.args.sr_share:
                    pick = 0
                else:
                    pick = k

                image_sr = model_list[pick](x_q, x_kv, cascade_images_list[:(k-self.args.stu_iter_b2)])
                image_sr = image_sr.to(self.device)
                cascade_images_list.append(image_sr)
                cascade_images = image_sr

            image_sr_branch2 = cascade_images_list[-1]
            image_sr = model_list[-1](image_sr_branch1, image_sr_branch2)
            image_sr = self.args.alpha * image_sr + (1 - self.args.alpha) * images_lr_psn[:, :3, :, :]
            metric_dict['psnr'].append(self.cal_psnr(image_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(image_sr, images_hr))

            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(image_sr[:, :3, :, :])
                moran_output = rec(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True, debug=True)
                preds, _ = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
                
                moran_input = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output = rec(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True, debug=True)
                preds, _ = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(image_sr[:, :3, :, :])
                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = rec(aster_dict_lr)
                aster_output_sr = rec(aster_dict_sr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(image_sr[:, :3, :, :])
                crnn_output = rec(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
                
                crnn_input = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output = rec(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_lr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
                        
            for pred, target in zip(pred_str_sr, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct += 1

            sum_images += val_batch_size
            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        print('[{}] | '
              'PSNR {:.2f} | SSIM {:.4f}'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg), float(ssim_avg)))
        print('save display images')

        self.tripple_display(images_lr, image_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)
        accuracy = round(n_correct / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        print('aster_accuray: %.2f%%' % (accuracy * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg
        return metric_dict

    def test(self):
        pygame.init()
        freetype.init()
        TP_Generator_dict = {
            "crnn": self.CRNN_init,
            "aster": self.Aster_init,
            "moran": self.MORAN_init,
            "visionlan": self.VisionLAN_init
        }

        stu_iter_b1 = self.args.stu_iter_b1
        stu_iter_b2 = self.args.stu_iter_b2

        recognizer_student = []
        test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                           cfgs.dataset_cfgs['case_sensitive'])
        for i in range(stu_iter_b1):
            recognizer_path = os.path.join(self.rec_path, "recognizer_best_" + str(i))
            if self.args.tpg=="aster":
                recognizer_path += ".pth.tar"
            else:
                recognizer_path += ".pth"
            recognizer_student_ = TP_Generator_dict[self.args.tpg](path=recognizer_path) 
            if type(recognizer_student_) == list:
                recognizer_student_ = recognizer_student_[i]
            for p in recognizer_student_.parameters():
                p.requires_grad = False
            recognizer_student_.eval()
            recognizer_student.append(recognizer_student_)

        model_psn = self.generator_init(0, psn=True)['model']
        for p in model_psn.parameters():
            p.requires_grad = False
        model_psn.eval()

        if self.args.arch == "tpgsr" or self.args.arch == "tatt":
            crnn_psn = self.CRNN_init(os.path.join(self.resume, "recognizer_best_crnn.pth"))
            crnn_psn.eval()
            for p in crnn_psn.parameters():
                p.requires_grad = False

        model_dict = self.generator_init(0, mode=False, hidden_size=3, testing=True)
        model, _ = model_dict['model'], model_dict['crit']
        model_list = [model]
        if not self.args.sr_share:
            for i in range(stu_iter_b1-1):
                model_sep = self.generator_init(i+1, mode=False, hidden_size=3, testing=True)['model']
                model_list.append(model_sep)

        model_list.append(self.generator_init(stu_iter_b1, mode=True, hidden_size=3, testing=True)['model'])
        if not self.args.sr_share:
            for i in range(stu_iter_b1, stu_iter_b1+stu_iter_b2-1):
                model_sep = self.generator_init(i+1, mode=True, hidden_size=3, testing=True)['model']
                model_list.append(model_sep)

        model_cmm = ComplementationModulationModule()
        if self.config.TRAIN.ngpu == 1:
            model_cmm.load_state_dict(
                torch.load(
                    os.path.join(self.resume, "model_best_cmm.pth")
                )['state_dict_G']
                )
        else:
            model_cmm.load_state_dict(
                {'module.' + k: v for k, v in torch.load(
                    os.path.join(self.resume, "model_best_cmm.pth")
                )['state_dict_G'].items()}
                )
        model_cmm = model_cmm.to(self.device)
        if self.config.TRAIN.ngpu > 1:
            model_cmm = torch.nn.DataParallel(model_cmm, device_ids=range(self.config.TRAIN.ngpu))
        model_list.append(model_cmm)
        
        result_path = os.path.join(self.config.TRAIN.ckpt_dir, "test_result.csv")
        _, test_loader = self.get_test_data(self.test_data_dir)
        data_name = self.args.test_data_dir.split('/')[-1]
        print('evaling %s' % data_name)
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        
        for model in model_list:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        time_begin = time.time()
        print('='*110)
        for j, data in (enumerate(test_loader)):
            images_hr, images_lr, _, _, label_strs, _ = data
            if self.args.rotate_test:
                batch_size = images_lr.shape[0]
                angle_batch = np.random.rand(batch_size) * self.args.rotate_train * 2 - self.args.rotate_train
                arc = angle_batch / 180. * math.pi
                rand_offs = torch.tensor(np.random.rand(batch_size)).float()
                arc = torch.tensor(arc).float()
                images_lr = torch_rotate_img(images_lr, arc, rand_offs)
                images_hr = torch_rotate_img(images_hr, arc, rand_offs)

            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)

            if self.args.arch in ['tsrn', 'tbsrn', 'tg']:
                images_lr_psn = model_psn(images_lr)
            elif self.args.arch == 'tpgsr':
                crnn_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                label_vecs_logits = crnn_psn(crnn_dict_lr)
                label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                images_lr_psn = model_psn(images_lr, label_vecs_final * 1.)
            elif self.args.arch == 'tatt':
                crnn_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                label_vecs_logits = crnn_psn(crnn_dict_lr)
                label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                images_lr_psn, _ = model_psn(images_lr, label_vecs_final.detach())

            cascade_images = images_lr_psn
            cascade_images_list = []

            for k in range(stu_iter_b1):
                x_q = torch.empty(1, 2, cascade_images.size(2), cascade_images.size(3))
                for i in range(cascade_images.size(0)):
                    visionlan_dict_lr = self.parse_visionlan_data(images_lr_psn[i, :3, :, :])
                    target = ''
                    label_lr, label_length = recognizer_student[k](visionlan_dict_lr, target, '', False)
                    pred_str_lr, _ = test_acc_counter.convert(label_lr, label_length)
                    s = pred_str_lr[0]
                    if s == "" or type(s) == torch.Tensor:
                        s = "\t"
                    lower_case = s.lower()
                    upper_case = s.upper()
                    i_t_lower = make_standard_text(self.args.font_path, lower_case, (cascade_images.size(2), cascade_images.size(3)))
                    i_t_lower_tensor = torch.from_numpy(i_t_lower).unsqueeze(0).unsqueeze(0)
                    i_t_upper = make_standard_text(self.args.font_path, upper_case, (cascade_images.size(2), cascade_images.size(3)))
                    i_t_upper_tensor = torch.from_numpy(i_t_upper).unsqueeze(0).unsqueeze(0)
                    i_t_tensor = torch.cat([i_t_lower_tensor, i_t_upper_tensor], dim=1)
                    x_q = torch.cat([x_q, i_t_tensor], dim=0)
                      
                x_q = x_q[1:]
                x_kv = cascade_images[:, :3, :]
                x_q = x_q.to(self.device)
                x_kv = x_kv.to(self.device)

                if self.args.sr_share:
                    pick = 0
                else:
                    pick = k

                image_sr = model_list[pick](x_q, x_kv, cascade_images_list[:k])
                image_sr = image_sr.to(self.device)
                cascade_images_list.append(image_sr)
                cascade_images = image_sr

            image_sr_branch1 = cascade_images_list[-1]
            cascade_images = images_lr_psn
            cascade_images_list = []
            for k in range(stu_iter_b1, stu_iter_b1+stu_iter_b2):
                x_q = torch.empty(1, 3, cascade_images.size(2), cascade_images.size(3))
                for i in range(cascade_images.size(0)):
                    mask = toMask(cascade_images[i][:3, :])
                    x_q = torch.cat([x_q, mask], dim=0)

                x_q = x_q[1:]
                x_kv = cascade_images[:, :3, :]
                x_q = x_q.to(self.device)
                x_kv = x_kv.to(self.device)

                if self.args.sr_share:
                    pick = 0
                else:
                    pick = k

                image_sr = model_list[pick](x_q, x_kv, cascade_images_list[:(k-self.args.stu_iter_b2)])
                image_sr = image_sr.to(self.device)
                cascade_images_list.append(image_sr)
                cascade_images = image_sr
                
            image_sr_branch2 = cascade_images_list[-1]
            images_sr = model_list[-1](image_sr_branch1, image_sr_branch2)
            images_sr = self.args.alpha * images_sr + (1 - self.args.alpha) * images_lr_psn[:, :3, :, :]
            images_sr = images_sr.to(self.device)
            val_batch_size = images_lr.shape[0]

            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, _ = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
                
                moran_input = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, _ = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
                
                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
                
                crnn_input = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_lr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(pred_str_sr, label_strs):
                pred_str = str_filt(pred, 'lower')
                true_str = str_filt(target, 'lower')
                if  pred_str == true_str:
                    n_correct += 1
            sum_images += val_batch_size
            torch.cuda.empty_cache()
            print('Evaluation: [{}][{} / {}]'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          j + 1, len(test_loader)))
            #self.test_display(images_lr[:, :3, :], images_sr[:, :3, :], images_hr[:, :3, :], pred_str_lr, pred_str_sr, label_strs, str_filt)
        time_end = time.time()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        duration = (time_end - time_begin) / sum_images
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'duration': duration}
        print(result)
        print('='*110)
        with open(result_path, "a+", newline="") as out:
            writer = csv.writer(out)
            writer.writerow([self.args.rec, data_name, acc, psnr_avg, ssim_avg])