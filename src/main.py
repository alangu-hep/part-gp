#!/usr/bin/env python3

import pysr
from pysr import PySRRegressor

import sys
import os
import argparse

from handlers import trainer, evaluation, annealing, sr_trainer
from handlers.args import setup_argparse

from ml_utils import losses
from ml_utils import surrogates
from ml_utils.optimizers import optim

from preprocessing.dataloaders import train_load, test_load
from preprocessing.datasets import SimpleIterDataset

from postprocessing.io_writer import _write_outputs_to_root

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh
plt.style.use(mh.style.CMS)

from importlib.util import spec_from_file_location, module_from_spec

from weaver.utils.logger import _logger, warn_n_times, _configLogger
import copy
from pprint import pformat
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def import_module(path, name='_mod'):
    pkg_root = os.path.dirname(os.path.abspath(path))
    not_inserted = pkg_root not in sys.path
    if not_inserted:
        sys.path.insert(0, pkg_root)

    spec = spec_from_file_location(name, path, 
        submodule_search_locations=[]
    )
    mod = module_from_spec(spec)
    mod.__package__ = name
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path.remove(pkg_root)
    return mod

def assemble_loaders(args):

    loaderdict = {
        'train': [],
        'val': [],
        'test': [],
    }

    if args.data_train and args.data_val:
        train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(args)
        loaderdict['train'] = train_loader
        loaderdict['val'] = val_loader
    if args.data_test:
        test_loaders, test_config = test_load(args)
        loaderdict['test'] = test_loaders

    return loaderdict

def initialize_models(args, training, network, model_path=None):
    
    network_module = import_module(network, name='_network_module')
    data_config = SimpleIterDataset({}, args.data_config, for_training=training).config
    model, model_info = network_module.get_model(data_config)

    if model_path is not None:
        wts = torch.load(model_path, map_location = 'cpu', weights_only = True)
        model.load_state_dict(wts)
    
    model_metadata = {
        'model': model,
        'info': model_info
    }

    return model_metadata

def save_logits(args, metrics_dict):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    logits = metrics_dict['preds']['logits']
    bkg = logits[:, 0]
    signal = logits[:, 1]
    confident = np.max(logits, axis=1)
    least_confident = np.min(logits, axis=1)
    ax[0, 0].hist(signal, bins=100)
    ax[0, 0].set_title('Signal Logit Distribution')
    ax[0, 0].set_xlabel('Value')
    ax[0, 0].set_ylabel('log(Counts)')

    ax[0, 1].hist(bkg, bins=100)
    ax[0, 1].set_title('Background Logit Distribution')
    ax[0, 1].set_xlabel('Value')
    ax[0, 1].set_ylabel('log(Counts)')

    ax[1, 0].hist(confident, bins=100)
    ax[1, 0].set_title('Most Confident Logit Distribution')
    ax[1, 0].set_xlabel('Value')
    ax[1, 0].set_ylabel('log(Counts)')

    ax[1, 1].hist(least_confident, bins=100)
    ax[1, 1].set_title('Least Confident Logit Distribution')
    ax[1, 1].set_xlabel('Value')
    ax[1, 1].set_ylabel('log(Counts)')

    for i in range(2):
        for j in range(2):
            ax[i, j].set_yscale('log')

    fig.savefig(f'{args.figures_prefix}_logits.png', dpi=300, bbox_inches='tight', transparent=True, format='png')

    plt.close(fig)

def classifier(args, loader_dict, model_dict):

    model = copy.deepcopy(model_dict['model']).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt, scheduler = optim(args, model, device)
    grad_scaler = torch.amp.GradScaler("cuda")

    model_name = args.model_prefix[args.model_prefix.rfind('/') + 1:]
    _logger.info(f'Model Name: {model_name}')

    model_trainer = trainer.SupervisedTrainer(
        loss_fn,
        model=model,
        opt=opt,
        scheduler=scheduler,
        train_loader=loader_dict['train'],
        device=device,
        grad_scaler=grad_scaler
    )

    _logger.info('Trainer Initialized')

    for epoch in range(args.num_epochs):
        _logger.info('-' * 50)
        _logger.info('Epoch #%d training' % epoch)
        model_trainer.train()

        if args.model_prefix:
            dirname = os.path.dirname(args.model_prefix)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            state_dict = model.module.state_dict() if isinstance(
                model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict()
            torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
            torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)

        _logger.info(f'Starting Validation for epoch {epoch}')
        model_val = evaluation.ClassificationStats(
            loss_fn,
            model=model,
            device=device,
            loader=loader_dict['val'],
            split='val'
        )
    
        _logger.info('Validator Initialized')
        val_dict = model_val.run()
        dict_print = pformat(val_dict, indent=2)
        _logger.info(f"Val Dict:\n{dict_print}")

        
    _logger.info(f'Testing Model')
    model_tester = evaluation.ClassificationStats(
        loss_fn,
        model=model,
        device=device,
        loader=loader_dict['test'],
        split='test'
    )
    _logger.info('Tester Initialized')
    test_dict = model_tester.run()
    dict_print = pformat(test_dict, indent=2)
    _logger.info(f"Test Dict:\n{dict_print}")
    #output_dir = args.metrics_prefix[:args.metrics_prefix.rfind('/')]
    #os.makedirs(output_dir, exist_ok=True)
    #_write_outputs_to_root(args.metrics_prefix, test_dict)

# remove this later

    _logger.info('Re-evaluating for training set logits')

    logit_tester = evaluation.ClassificationStats(
        loss_fn,
        model=model,
        device=device,
        loader=loader_dict['train'],
        split='test'
    )

    _logger.info('Train Set Tester Initialized')
    logit_dict = logit_tester.run()

    save_logits(args, logit_dict)

def knowledge_distillation(args, loader_dict, teacher_dict, student_dict):
    student = copy.deepcopy(student_dict['model']).to(device)
    teacher = copy.deepcopy(teacher_dict['model']).to(device)
    student_name = args.model_prefix[args.model_prefix.rfind('/') + 1:]
    teacher_name = args.teacher_prefix
    _logger.info(f'Teacher Name: {teacher_name}')
    _logger.info(f'Student Name: {student_name}')
    
    opt, scheduler = optim(args, student, device)
    grad_scaler = torch.amp.GradScaler("cuda")
    
    if args.kd_anneal:
        _logger.info('Using KD Temperature Annealer')
        steps = int((1e+08 * (0.2)) * args.data_fraction)
        annealer = annealing.Annealer(
            total_steps=steps,
            shape='cosine',
            baseline=(1 / args.kd_temp),
            cyclical=False
        )

    loss_fn = nn.CrossEntropyLoss()
    kd_loss = losses.KD_DKL(
        T=args.kd_temp,
        temp_annealer=annealer if args.kd_anneal else None,
        reduction='batchmean',
        weight=args.kd_weight
    )

    model_trainer = trainer.KDTrainer(
        teacher=teacher,
        cl_loss = loss_fn,
        kd_loss = kd_loss,
        model=student,
        opt=opt,
        scheduler=scheduler,
        train_loader=loader_dict['train'],
        device=device,
        grad_scaler=grad_scaler
    )

    _logger.info('Trainer Initialized')
    
    for epoch in range(args.num_epochs):
        _logger.info('-' * 50)
        _logger.info('Epoch #%d training' % epoch)
        model_trainer.train()

        if args.model_prefix:
            dirname = os.path.dirname(args.model_prefix)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            state_dict = student.module.state_dict() if isinstance(
                student, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else student.state_dict()
            torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
            torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)

        _logger.info(f'Starting Validation for epoch {epoch}')
        model_val = evaluation.KDStats(
            teacher=teacher,
            cl_loss=loss_fn,
            kd_loss=kd_loss,
            model=student,
            device=device,
            loader=loader_dict['val'],
            split='val'
        )
        _logger.info('Validator Initialized')
        val_dict = model_val.run()
        dict_print = pformat(val_dict, indent=2)
        _logger.info(f"Val Dict:\n{dict_print}")
        
    _logger.info(f'Testing Model')
    model_tester = evaluation.KDStats(
        teacher=teacher,
        cl_loss=loss_fn,
        kd_loss=kd_loss,
        model=student,
        device=device,
        loader=loader_dict['test'],
        split='test'
    )
    _logger.info('Tester Initialized')
    test_dict = model_tester.run()
    dict_print = pformat(test_dict, indent=2)
    _logger.info(f"Test Dict:\n{dict_print}")
    #output_dir = args.metrics_prefix[:args.metrics_prefix.rfind('/')]
    #os.makedirs(output_dir, exist_ok=True)
    #_write_outputs_to_root(args.metrics_prefix, test_dict)

    _logger.info('Re-evaluating for training set logits')

    logit_tester = evaluation.KDStats(
        teacher=teacher,
        cl_loss=loss_fn,
        kd_loss=kd_loss,
        model=student,
        device=device,
        loader=loader_dict['train'],
        split='test'
    )

    _logger.info('Train Set Tester Initialized')
    logit_dict = logit_tester.run()

    save_logits(args, logit_dict)

def svae(args, loader_dict, vae_dict, teacher_dict):

    vae = copy.deepcopy(vae_dict['model']).to(device)
    teacher = copy.deepcopy(teacher_dict['model']).to(device)
    opt, scheduler = optim(args, vae, device)
    grad_scaler = torch.amp.GradScaler("cuda")

    train_size = (1e+08 * (0.2)) * args.data_fraction
    val_size = (5e+06 * (0.2)) * args.data_fraction
    test_size = (2e+07 * (0.2)) * args.data_fraction

    if args.kl_anneal:
        _logger.info('Using KL Annealer')
        steps = int(train_size)
        annealer = annealing.Annealer(
            total_steps=steps,
            shape='cosine',
            baseline=0,
            r_value=0.5,
            cyclical=True
        )

    vae_loss = losses.TCVAELoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        use_mss=True,
        annealer=annealer if args.kl_anneal else None
    )

# potentially implement supervised annealer?
    
    reg_loss = losses.HuberLoss(
        threshold=1.35,
        weight=args.svae_weight,
        annealer=None
    )

    recon_loss = losses.ChamferDist()

    model_trainer = trainer.STCVAETrainer(
        teacher=teacher,
        sup_loss=reg_loss,
        vae_loss = vae_loss,
        recon_loss = recon_loss,
        dataset_size = train_size,
        model=vae,
        opt=opt,
        scheduler=scheduler,
        train_loader=loader_dict['train'],
        device=device,
        grad_scaler=grad_scaler,
        clip_norm=1.0
    )

    _logger.info('Trainer Initialized')

    for epoch in range(args.num_epochs):
        _logger.info('-' * 50)
        _logger.info('Epoch #%d training' % epoch)
        model_trainer.train()

        if args.model_prefix:
            dirname = os.path.dirname(args.model_prefix)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            state_dict = vae.module.state_dict() if isinstance(
                vae, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else vae.state_dict()
            torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
            torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)

        _logger.info(f'Starting Validation for epoch {epoch}')
        
        model_val=evaluation.STCVAEStats(
            teacher=teacher,
            sup_loss=reg_loss,
            vae_loss=vae_loss,
            recon_loss=recon_loss,
            dataset_size=val_size,
            model=vae,
            device=device,
            loader=loader_dict['val'],
            split='val'
        )
        
        _logger.info('Validator Initialized')
        val_dict = model_val.run()
        _logger.info(f"Latent Variances for epoch {epoch}: {val_dict['interpretability']['latent_variances']}")

    _logger.info(f'Testing Model')
    
    model_tester = evaluation.STCVAEStats(
        teacher=teacher,
        sup_loss=reg_loss,
        vae_loss = vae_loss,
        recon_loss = recon_loss,
        dataset_size=test_size,
        model=vae,
        device=device,
        loader=loader_dict['test'],
        split='test'
    )

    _logger.info('Tester Initialized')
    test_dict = model_tester.run()
    dict_print = pformat(test_dict, indent=2)
    _logger.info(f"Test Dict:\n{dict_print}")

def symbolic_regression(args, loader_dict, model_dict, vae_dict):

    model = copy.deepcopy(model_dict['model']).to(device)
    dr = copy.deepcopy(vae_dict['model']).to(device)

    model_trainer = sr_trainer.SymbolicTrainer(model, dr, loader_dict['train'], device)
    regressor = model_trainer.run(args)

    _logger.info('Regressor Params:' + str(regressor.get_params()))

    modules = regressor.pytorch()
    surrogate = surrogates.Surrogate(modules)

    loss_fn = torch.nn.CrossEntropyLoss()

    model_tester = evaluation.SurrogateStats(
        dr=dr,
        loss=loss_fn,
        model=surrogate,
        device=device,
        loader=loader_dict['test'],
        split='test'
    )
    
    _logger.info('Tester Initialized')
    test_dict = model_tester.run()
    dict_print = pformat(test_dict, indent=2)
    _logger.info(f"Test Dict:\n{dict_print}")

def main():

    args = setup_argparse().parse_args()

    stdout = sys.stdout
    _configLogger('weaver', stdout=stdout, filename=args.log)
    _logger.info('Started!')
    _logger.info('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))
    
    loader_dict = assemble_loaders(args)

    start = time.time()
    
    if args.comp == 'DL':
        _logger.info('Training Regular Classifier')
        model_dict = initialize_models(args, training=True, network=args.model_network)
        classifier(args, loader_dict, model_dict)
    elif args.comp == 'KD':
        _logger.info(f'Performing DL Knowledge Distillation w/ teacher at path {args.teacher_path}')
        _logger.info(f'KD Temperature: {args.kd_temp}')
        student_dict = initialize_models(args, training=True, network=args.model_network)
        teacher_dict = initialize_models(args, training=True, network=args.teacher_network, model_path=args.teacher_path)
        knowledge_distillation(args, loader_dict, teacher_dict, student_dict)
    elif args.comp == 'SDR':
        _logger.info('Performing Supervised Dimensionality Reduction')
        vae_dict = initialize_models(args, training=True, network=args.dr_network)
        teacher_dict = initialize_models(args, training=True, network=args.model_network, model_path=args.model_path)
        svae(args, loader_dict, vae_dict, teacher_dict)
    elif args.comp == 'SR':
        _logger.info('Performing Symbolic Regression')
        vae_dict = initialize_models(args, training=True, network=args.dr_network, model_path=args.dr_path)
        model_dict = initialize_models(args, training=True, network=args.model_network, model_path=args.model_path)
        symbolic_regression(args, loader_dict, model_dict, vae_dict)

    end = time.time()
    elapsed_time = end - start

    _logger.info(f'Elapsed time: {elapsed_time:.2f} seconds')

if __name__ == '__main__':
    main()