import torch

import sys
import os

from weaver.utils.logger import _logger, warn_n_times
from collections import defaultdict, Counter
from ml_utils import losses
from ml_utils.tracker import AverageTracker
from ml_utils.flattening import _flatten_label, _flatten_preds

import tqdm
import time

class BaseTrainer:
    def __init__(self, model, opt, scheduler, train_loader, device, grad_scaler=None, clip_norm = None):

        model.train()
        self.model = model
        
        self.opt = opt
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        
        self.train_loader = train_loader
        self.device = device
        
        self.clip_norm = clip_norm

        self.data_config = train_loader.dataset.config
        self.avg_trackers = defaultdict(AverageTracker)
        
    def compute_loss(self, inputs, mask, label):
        raise NotImplementedError
        
    def train(self):
        with tqdm.tqdm(self.train_loader) as tq:
            for X, y, _ in tq:
                inputs = [X[k].to(self.device) for k in self.data_config.input_names]
                label = y[self.data_config.label_names[0]].long().to(self.device)
                try:
                    mask = y[self.data_config.label_names[0] + '_mask'].bool().to(self.device)
                except KeyError:
                    mask = None
                self.opt.zero_grad()
                
                with torch.amp.autocast("cuda", enabled=self.grad_scaler is not None):
                    loss, extra_display = self.compute_loss(inputs, label, mask)

                self.backprop(loss)
                loss = loss.item()

                tq.set_postfix({
                    'lr': '%.2e' % self.scheduler.get_last_lr()[0] if self.scheduler else self.opt.defaults['lr'],
                    'Loss': '%.5f' % loss,
                    **extra_display,
                    **{k: f'{v:.5f}' for k, v in self.fetch_means().items()}
                })
                
    def backprop(self, loss):
        if self.grad_scaler is None:
            loss.backward()
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.opt.step()
        else:
            self.grad_scaler.scale(loss).backward()
            if self.clip_norm is not None:
                self.grad_scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
        if self.scheduler and getattr(self.scheduler, '_update_per_step', False):
            self.scheduler.step()
    def track_avg(self, metrics: dict[str, float]):
        for key, value in metrics.items():
            self.avg_trackers[key].update(value)
    def fetch_means(self) -> dict[str, float]:
        return {k: v.mean for k, v in self.avg_trackers.items()}

class SupervisedTrainer(BaseTrainer):
    def __init__(self, base_loss, **kwargs):
        super(SupervisedTrainer, self).__init__(**kwargs)
        self.base_loss = base_loss
        
    def compute_loss(self, inputs, label, mask):
        model_output = self.model(*inputs)
        logits, labels, _ = _flatten_preds(model_output, label=label, mask=mask)
        loss = self.base_loss(logits, label)

        _, preds = logits.max(1)
        correct = (preds == label).sum().item()
        num_examples = label.shape[0]

        self.track_avg({'AvgAcc': (correct/num_examples)})

        extra_display = {
            'Acc': '%.5f' % (correct / num_examples),
        }
        return loss, extra_display

class KDTrainer(BaseTrainer):
    def __init__(self, teacher, cl_loss, kd_loss, **kwargs):
        super(KDTrainer, self).__init__(**kwargs)
        teacher.eval()
        self.teacher = teacher
        self.cl_loss = cl_loss
        self.kd_loss = kd_loss

    def compute_loss(self, inputs, label, mask):

        with torch.no_grad():
            teacher_output = self.teacher(*inputs)
        student_output = self.model(*inputs)
        
        t_logits, t_label, _ = _flatten_preds(teacher_output, label=label, mask=mask)
        s_logits, s_label, _ = _flatten_preds(student_output, label=label, mask=mask)
        
        class_loss = self.cl_loss(s_logits, label)
        kd_loss = self.kd_loss(s_logits, t_logits)
        loss = class_loss + kd_loss

        _, preds = s_logits.max(1)
        correct = (preds == label).sum().item()
        num_examples = label.shape[0]

        class_loss = class_loss.item()
        kd_loss = kd_loss.item()

        self.track_avg({'AvgCL': class_loss, 'AvgKDL': kd_loss, 'AvgAcc': (correct/num_examples)})

        extra_display = {
            'Acc': '%.5f' % (correct / num_examples),
            'Class Loss': '%.5f' % class_loss,
            'KD Loss': '%.5f' % kd_loss
        }
        return loss, extra_display

class TCVAETrainer(BaseTrainer):
    def __init__(self, vae_loss, recon_loss, dataset_size, **kwargs):
        super(TCVAETrainer, self).__init__(**kwargs)
        self.vae_loss = vae_loss
        self.recon_loss = recon_loss
        self.data_size = dataset_size

    def compute_loss(self, inputs, label, mask):
        reconstructed, mean, log_var, z = self.model(*inputs)
        target = inputs[1]
        rec_mask = inputs[3].squeeze(1)
        mi_loss, tc_loss, dw_kl_loss, alpha, beta, gamma = self.vae_loss(z, (mean, log_var), self.data_size)
        recon_loss = self.recon_loss(reconstructed, target, rec_mask)
        loss = alpha*mi_loss + beta*tc_loss + gamma*dw_kl_loss + recon_loss

        mi_loss = mi_loss.item()
        tc_loss = tc_loss.item()
        dw_kl_loss = dw_kl_loss.item()
        recon_loss = recon_loss.item()

        self.track_avg({
            'Avg MIL': mi_loss,
            'Avg TCL': tc_loss,
            'Avg DWKL': dw_kl_loss,
            'Avg Recon Loss': recon_loss,
        })

        extra_display = {
            'MIL': '%.5f' % mi_loss,
            'TCL': '%.5f' % tc_loss,
            'DWKL': '%.5f' % dw_kl_loss,
            'Recon Loss': '%.5f' % recon_loss,
            'Beta': '%.2f' % beta
        }
        
        return loss, extra_display

class STCVAETrainer(TCVAETrainer):
    def __init__(self, teacher, sup_loss, **kwargs):
        super(STCVAETrainer, self).__init__(**kwargs)
        teacher.eval()
        self.teacher = teacher
        self.sup_loss = sup_loss

    def compute_loss(self, inputs, label, mask):
        with torch.no_grad():
            teacher_logits = self.teacher(*inputs)
        reconstructed, mean, log_var, z, logits = self.model(*inputs)
        target = inputs[1]
        rec_mask = inputs[3].squeeze(1)
        mi_loss, tc_loss, dw_kl_loss, alpha, beta, gamma = self.vae_loss(z, (mean, log_var), self.data_size)
        recon_loss = self.recon_loss(reconstructed, target, rec_mask)
        #sup_loss, sup_wt = self.sup_loss(logits, teacher_logits)
        loss = alpha*mi_loss + beta*tc_loss + gamma*dw_kl_loss + recon_loss# + sup_wt*sup_loss

        mi_loss = mi_loss.item()
        tc_loss = tc_loss.item()
        dw_kl_loss = dw_kl_loss.item()
        recon_loss = recon_loss.item()
        #sup_loss = sup_loss.item()

        self.track_avg({
            'AvgMIL': mi_loss,
            'AvgTCL': tc_loss,
            'AvgDWKL': dw_kl_loss,
            'AvgReconLoss': recon_loss
            #'AvgSupLoss': sup_loss
        })
        
        extra_display = {
            'MIL': '%.5f' % mi_loss,
            'TCL': '%.5f' % tc_loss,
            'DWKL': '%.5f' % dw_kl_loss,
            'ReconLoss': '%.5f' % recon_loss,
            #'SupLoss': '%.5f' % sup_loss,
            'Beta': '%.2f' % beta
        }

        return loss, extra_display