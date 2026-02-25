import sys
import os

from weaver.utils.logger import _logger, warn_n_times
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from weaver.utils.data.tools import _concat
import torch
from weaver.train import to_filelist
from src.preprocessing.datasets import SimpleIterDataset
import functools

import numpy as np
import awkward as ak
import tqdm
import time
import math

def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label


def _flatten_preds(model_output, label=None, mask=None, label_axis=1):
    if not isinstance(model_output, tuple):
        # `label` and `mask` are provided as function arguments
        preds = model_output
    else:
        if len(model_output == 2):
            # use `mask` from model_output instead
            # `label` still provided as function argument
            preds, mask = model_output
        elif len(model_output == 3):
            # use `label` and `mask` from model output
            preds, label, mask = model_output

    # preds: (N, num_classes); (N, num_classes, P)
    # label: (N,);             (N, P)
    # mask:  None;             (N, P) / (N, 1, P)
    if preds.ndim > 2:
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)

    if label is not None:
        label = _flatten_label(label, mask)

    return preds, label, mask


def knowledge_distillation(
        teacher, student, loss_func, opt, scheduler, train_loader, dev, epoch, T=1.0, steps_per_epoch=None, class_weight=1.0, kl_weight=0.1, grad_scaler=None,
        tb_helper=None):
    teacher.eval()
    student.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev)
            entry_count += label.shape[0]
            try:
                mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
            except KeyError:
                mask = None
            opt.zero_grad()
            kl_div = torch.nn.KLDivLoss(reduction='batchmean')
            
            with torch.amp.autocast("cuda", enabled=grad_scaler is not None):
                with torch.no_grad():
                    teacher_output = teacher(*inputs)
                t_logits, t_label, _ = _flatten_preds(teacher_output, label=label, mask=mask)
                t_softmax = torch.nn.functional.softmax(t_logits/T, dim=-1)
                student_output = student(*inputs)
                s_logits, s_label, _ = _flatten_preds(student_output, label=label, mask=mask)
                s_softmax = torch.nn.functional.log_softmax(s_logits/T, dim=-1)
                loss = class_weight*loss_func(s_logits, label) + kl_weight*kl_div(s_softmax, t_softmax)*(T*T)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds = s_logits.max(1)
            loss = loss.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=student_output, model=student,
                                            epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=student_output, model=student, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()

def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    B = batch_size

    strat_weight = (N - B) / (N * (B - 1))

    W = torch.full((B, B), strat_weight)
    diag = torch.arange(B)
    W[diag, diag] = 1.0 / N

    return W.log()


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        mat_log_qz_sum = mat_log_qz.sum(2)
        
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz_sum = mat_log_qz_sum + log_iw_mat
        
        log_qz = torch.logsumexp(mat_log_qz_sum, dim=1, keepdim=False)
        
        mat_log_qz_weighted = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)
        log_prod_qzi = torch.logsumexp(mat_log_qz_weighted, dim=1, keepdim=False).sum(1)
    else:
        log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
        log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

def tcvae_loss(latent_sample, latent_dist, n_data, is_mss=True):

    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                         latent_dist,
                                                                         n_data,
                                                                         is_mss)
    # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    mi_loss = (log_q_zCx - log_qz).mean()
    # TC[z] = KL[q(z)||\prod_i z_i]
    tc_loss = (log_qz - log_prod_qzi).mean()
    # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
    dw_kl_loss = (log_prod_qzi - log_pz).mean()

    return mi_loss, tc_loss, dw_kl_loss
    

def chamfer_loss(predictions, targets, mask, squared=True):

    batch_losses = []
    
    for i in range(predictions.size(0)):
        valid_mask = mask[i].bool() 
        
        pred_valid = predictions[i, :, valid_mask]
        target_valid = targets[i, :, valid_mask] 
        
        pred_valid = pred_valid.T
        target_valid = target_valid.T
        
        dist = torch.cdist(pred_valid, target_valid, p=2)
        
        if squared:
            dist = dist.pow(2)
        
        loss = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
        batch_losses.append(loss)
    
    return torch.stack(batch_losses).mean()
    
def train_autoencoder(model, opt, scheduler, train_loader, dev, epoch, dataset_size, alpha=1.0, beta=6.0, gamma=1.0, steps_per_epoch=None, grad_scaler=None, tb_helper=None, annealer=None, max_norm=1.0):
    
    model.train()
    data_config = train_loader.dataset.config
    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev)
            entry_count += label.shape[0]
            if annealer is not None:
                beta_t = beta * annealer()
                annealer.step()
            else:
                beta_t = beta
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=grad_scaler is not None):
                reconstructed, mean, log_var, z = model(*inputs)
                target = inputs[1]
                rec_mask = inputs[3].squeeze(1)
                mi_loss, tc_loss, dw_kl_loss = tcvae_loss(z, (mean, log_var), dataset_size, is_mss=True)
                reconstruction_loss = chamfer_loss(reconstructed, target, rec_mask, squared=True)
                loss = alpha*mi_loss + beta_t*tc_loss + gamma*dw_kl_loss + reconstruction_loss
            if grad_scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            loss = loss.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            total_loss += loss

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'MIL': '%.5f' % (mi_loss.item()),
                'TCL': '%.5f' % (tc_loss.item()),
                'DWKL': '%.5f' % (dw_kl_loss.item()),
                'Reconstruction Loss': '%.5f' % reconstruction_loss.item(),
                'Beta': '%.2f' % beta_t
            })

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()

def evaluate_autoencoder(model, test_loader, dev, epoch, dataset_size, alpha=1.0, beta=6.0, gamma=1.0, for_training=True, loss_func=None, steps_per_epoch=None, tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_mil = 0
    total_tcl = 0
    total_dwkl = 0
    total_rec = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    latents = []
    outputs = []
    input_data = []
    labels = []
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                labels.append(label)
                try:
                    mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
                except KeyError:
                    mask = None
                reconstructed, mean, log_var, z = model(*inputs)
                target = inputs[1]
                rec_mask = inputs[3].squeeze(1)
                mi_loss, tc_loss, dw_kl_loss = tcvae_loss(z, (mean, log_var), dataset_size, is_mss=True)
                reconstruction_loss = chamfer_loss(reconstructed, target, rec_mask, squared=True)
                loss = alpha*mi_loss + beta*tc_loss + gamma*dw_kl_loss + reconstruction_loss
                latents.append(torch.cat([mean, log_var], axis=1))
                input_data.append(target)
                outputs.append(reconstructed)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))

                mask_sq = inputs[3].squeeze(1)
                n = mask_sq.sum(dim=1).long()
                test_mean, test_var, perm = model.mod.encoder(target, n)
                print(perm.shape)
                print(perm)
                
                z_rand = torch.randn_like(z)
                x_from_perm = model.mod.decoder(z_rand, perm, n)

                perm_rand = torch.rand_like(perm)
                perm_rand = perm_rand / perm_rand.sum(dim=-1, keepdim=True)

                x_from_z = model.mod.decoder(z, perm_rand, n)

                recon_rand_z = chamfer_loss(x_from_perm, target, rec_mask, squared=True)
                recon_rand_perm = chamfer_loss(x_from_z, target, rec_mask, squared=True)

                print(f'Random Z: {recon_rand_z.item()}')
                print(f'Random Perm: {recon_rand_perm.item()}')

                num_batches += 1
                count += num_examples
                total_loss += loss
                total_mil += mi_loss
                total_tcl += tc_loss
                total_dwkl += dw_kl_loss
                total_rec += reconstruction_loss

                tq.set_postfix({
                    'Loss': '%.5f' % loss, # weighted loss
                    'MIL': '%.5f' % (mi_loss.item()),
                    'TCL': '%.5f' % (tc_loss.item()),
                    'DWKL': '%.5f' % (dw_kl_loss.item()),
                    'Reconstruction Loss': '%.5f' % reconstruction_loss.item(),
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'AvgMIL': '%.5f' % (total_mil / num_batches),
                    'AvgTCL': '%.5f' % (total_tcl / num_batches),
                    'AvgDWKL': '%.5f' % (total_dwkl / num_batches),
                    'AvgREC': '%.5f' % (total_rec / num_batches)
                })


                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    return input_data, outputs, latents, labels

def prepare_logits(model, teacher, test_loader, dev, epoch, dataset_size, for_training=True, loss_func=None, steps_per_epoch=None,
                            tb_helper=None):
    model.eval()
    teacher.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_mil = 0
    total_tcl = 0
    total_dwkl = 0
    total_rec = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    outputs = []
    logits = []
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                try:
                    mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
                except KeyError:
                    mask = None
                reconstructed, mean, log_var, z = model(*inputs)
                target = inputs[1]
                rec_mask = inputs[3].squeeze(1)
                mi_loss, tc_loss, dw_kl_loss = tcvae_loss(z, (mean, log_var), dataset_size, is_mss=True)
                reconstruction_loss = chamfer_loss(reconstructed, target, rec_mask, squared=True)
                loss = mi_loss + tc_loss + dw_kl_loss + reconstruction_loss
                teacher_output = teacher(*inputs)
                t_logits, t_label, _ = _flatten_preds(teacher_output, label=label, mask=mask)
                outputs.append(torch.cat([mean, log_var], axis=1))
                logits.append(t_logits)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))

                num_batches += 1
                count += num_examples
                total_loss += loss
                total_mil += mi_loss
                total_tcl += tc_loss
                total_dwkl += dw_kl_loss
                total_rec += reconstruction_loss

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'MIL': '%.5f' % (mi_loss.item()),
                    'TCL': '%.5f' % (tc_loss.item()),
                    'DWKL': '%.5f' % (dw_kl_loss.item()),
                    'Reconstruction Loss': '%.5f' % reconstruction_loss.item(),
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'AvgMIL': '%.5f' % (total_mil / num_batches),
                    'AvgTCL': '%.5f' % (total_tcl / num_batches),
                    'AvgDWKL': '%.5f' % (total_dwkl / num_batches),
                    'AvgREC': '%.5f' % (total_rec / num_batches)
                })


                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    return outputs, logits

def final_run(teacher, dl_model, surrogate, loss_func, opt, scheduler, train_loader, dev, epoch, dataset_size, T=1.0, alpha=1.0, beta=6.0, gamma=1.0, steps_per_epoch=None, class_weight=1.0, kl_weight=0.1, grad_scaler=None, tb_helper=None, annealer = None):
    teacher.eval()
    dl_model.eval()
    surrogate.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_mil = 0
    total_tcl = 0
    total_dwkl = 0
    total_rec = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev)
            entry_count += label.shape[0]
            try:
                mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
            except KeyError:
                mask = None

            if annealer is not None:
                beta_t = beta * annealer()
                annealer.step()
            else:
                beta_t = beta
            opt.zero_grad()
            kl_div = torch.nn.KLDivLoss(reduction='batchmean')
            
            with torch.amp.autocast("cuda", enabled=grad_scaler is not None):
                with torch.no_grad():
                    teacher_output = teacher(*inputs)
                    dl_output = dl_model(*inputs)
                    
                t_logits, t_label, _ = _flatten_preds(teacher_output, label=label, mask=mask)
                t_softmax = torch.nn.functional.softmax(t_logits/T, dim=-1)
                
                d_logits, d_label, _ = _flatten_preds(dl_output, label=label, mask=mask)
                dl_softmax = torch.nn.functional.softmax(d_logits/T, dim=-1)
                
                surrogate_output, recon, mean, log_var, z = surrogate(*inputs)
                s_logits, s_label, _ = _flatten_preds(surrogate_output, label=label, mask=mask)
                s_softmax = torch.nn.functional.log_softmax(s_logits/T, dim=-1)

                target = inputs[1]
                rec_mask = inputs[3].squeeze(1)
                mi_loss, tc_loss, dw_kl_loss = tcvae_loss(z, (mean, log_var), dataset_size, is_mss=True)
                reconstruction_loss = chamfer_loss(recon, target, rec_mask, squared=True)
                
                loss = class_weight*loss_func(s_logits, label) + kl_weight*kl_div(s_softmax, t_softmax)*(T*T) + 1.0*kl_div(s_softmax, dl_softmax)+alpha*mi_loss + beta_t*tc_loss + gamma*dw_kl_loss + reconstruction_loss
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds = s_logits.max(1)
            loss = loss.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct
            total_mil += mi_loss.item()
            total_tcl += tc_loss.item()
            total_dwkl += dw_kl_loss.item()
            total_rec += reconstruction_loss.item()

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count),
                'MIL': '%.5f' % (mi_loss.item()),
                'TCL': '%.5f' % (tc_loss.item()),
                'DWKL': '%.5f' % (dw_kl_loss.item()),
                'Reconstruction Loss': '%.5f' % reconstruction_loss.item(),
                'AvgMIL': '%.5f' % (total_mil / num_batches),
                'AvgTCL': '%.5f' % (total_tcl / num_batches),
                'AvgDWKL': '%.5f' % (total_dwkl / num_batches),
                'AvgREC': '%.5f' % (total_rec / num_batches)
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=student_output, model=student,
                                            epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=student_output, model=student, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()

def evaluate_surrogates(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                try:
                    mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
                except KeyError:
                    mask = None
                model_output, recon, mean, log_var, z = model(*inputs)
                logits, label, mask = _flatten_preds(model_output, label=label, mask=mask)
                scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))

                if mask is not None:
                    mask = mask.cpu()
                for k, v in y.items():
                    labels[k].append(_flatten_label(v, mask).numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))
                if not for_training and mask is not None:
                    labels_counts.append(np.squeeze(mask.numpy(force=True).sum(axis=-1)))

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch,
                                                i_batch=num_batches, mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}

    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert (count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers