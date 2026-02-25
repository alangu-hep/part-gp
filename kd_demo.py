#!/usr/bin/env python3

import sys
import os

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import uproot
import awkward as ak
import copy
import shutil
import vector

from weaver.utils.import_tools import import_module
from src.preprocessing.datasets import SimpleIterDataset
from weaver.utils.nn.tools import evaluate_classification, train_classification
from weaver.utils.logger import _logger, warn_n_times

from src.part_prediction import test_load, train_load, knowledge_distillation, optim
from src.handlers.hook_handler import HookHandler

from src.metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_datasets = os.getenv("PART_DATA")

class Args:
    def __init__(self, **kwargs):
        # defaults
        self.data_train = []
        self.data_test = []
        self.data_val = []
        self.num_workers = 0
        self.num_epochs = 0
        self.data_config = ''
        self.extra_selection = None
        self.extra_test_selection = None
        self.file_fraction = 1
        self.data_fraction = 1
        self.fetch_by_files = False
        self.fetch_step = 0.01
        self.batch_size = 0
        self.in_memory = False
        self.local_rank = None
        self.copy_inputs = False
        self.demo = False
        self.no_remake_weights = False
        self.steps_per_epoch = None
        self.steps_per_epoch_val = None
        self.backend = None
        self.model_prefix = None
        self.lr_finder = None
        self.optimizer_option = []
        self.optimizer = 'ranger'
        self.start_lr = 1e-3
        self.final_lr = 1e-6
        self.lr_scheduler = 'flat+decay'
        self.load_epoch = None
        self.gpus = 0
        self.predict_gpus = 0
        self.regression_mode = False
        self.kl_weight = 0.1
        self.class_weight = 1.0
        
        for key, value in kwargs.items():
            setattr(self, key, value)

yaml_config = 'data_config/JetClass/JetClass_kin.yaml'
network_path = 'models/networks/part_wrapper.py'
teacher_path = 'models/networks/pelican_wrapper.py'

jc_paths = {
    'train': path_datasets+'/JetClass/Pythia/train_100M',
    'val': path_datasets+'/JetClass/Pythia/val_5M',
    'test': path_datasets+'/JetClass/Pythia/test_20M'
}

num_classes = 2
signal = '/TTBar_*.root'
background = '/ZJetsToNuNu_*.root'

datasets = {}

for name, path in jc_paths.items():

    if isinstance(signal, str):
        signal_files = glob.glob(path+signal)

    if isinstance(background, str):
        background_files = glob.glob(path+background)

    datasets[name] = signal_files + background_files

def initialize_CNN(network_path, config_path, training = True, model_path = None) -> dict:

    models = {}
    
    network_module = import_module(network_path, name='_network_module')
    data_config = SimpleIterDataset({}, config_path, for_training=training).config
    model, model_info = network_module.get_model(data_config)

    if model_path is not None:
        wts = torch.load(model_path, map_location = 'cpu', weights_only = True)
        model.load_state_dict(wts)
    
    model_metadata = {
        'model': model,
        'info': model_info,
        'loss': network_module.get_loss(data_config)
    }

    return model_metadata

def initialize_PELICAN(network_path, config_path, training = True, model_path = None) -> dict:

    models = {}
    
    network_module = import_module(network_path, name='_network_module')
    data_config = SimpleIterDataset({}, config_path, for_training=training).config
    model, model_info = network_module.get_model(data_config,
                                                 dataset='',
                                                 method='spurions',
                                                 stabilizer='so2',
                                                 average_nobj=32,
                                                 scale=0.1,
                                                 num_channels_scalar=10,
                                                 num_channels_m=[[60],]*5,
                                                 num_channels_2to2=[35,]*5,
                                                 num_channels_out=[60],
                                                 num_channels_m_out=[60, 35]
                                                )

    if model_path is not None:
        wts = torch.load(model_path, map_location = 'cpu', weights_only = True)
        model.load_state_dict(wts)
    
    model_metadata = {
        'model': model,
        'info': model_info,
        'loss': network_module.get_loss(data_config)
    }

    return model_metadata

complexity = {
    'particle_attn': 2,
    'class_attn': 1
}

def initialize_models(network_path, config_path, training = True, model_path = None) -> dict:

    models = {}
    
    network_module = import_module(network_path, name='_network_module')
    data_config = SimpleIterDataset({}, config_path, for_training=training).config
    model, model_info = network_module.get_model(data_config, num_layers = complexity['particle_attn'], num_cls_layers = complexity['class_attn'])

    if model_path is not None:
        wts = torch.load(model_path, map_location = 'cpu', weights_only = True)
        model.load_state_dict(wts)
    
    model_metadata = {
        'model': model,
        'info': model_info,
        'loss': network_module.get_loss(data_config)
    }

    return model_metadata

'''
metadata = initialize_CNN(network_path, yaml_config, training=True)

args = Args(
    data_train = datasets['train'],
    data_val = datasets['val'],
    data_test = datasets['test'],
    data_config = yaml_config,
    num_epochs = 5,
    batch_size = 64,
    model_prefix = 'models/torch_saved/ResNet',
    file_fraction = 1,
    data_fraction = 0.1
)

train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(args)

model = copy.deepcopy(metadata['model']).to(device)
loss_func = metadata['loss']

opt, scheduler = optim(args, model, device)

tb = None

# regular
best_valid_metric = np.inf if args.regression_mode else 0
grad_scaler = torch.amp.GradScaler("cuda")
for epoch in range(args.num_epochs):
    if args.load_epoch is not None:
        if epoch <= args.load_epoch:
            continue
    _logger.info('-' * 50)
    _logger.info('Epoch #%d training' % epoch)
    train_classification(model, loss_func, opt, scheduler, train_loader, device, epoch,
          steps_per_epoch=args.steps_per_epoch, grad_scaler=grad_scaler, tb_helper=tb)
    if args.model_prefix and (args.backend is None or local_rank == 0):
        dirname = os.path.dirname(args.model_prefix)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        state_dict = model.module.state_dict() if isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict()
        torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
        torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)
    # if args.backend is not None and local_rank == 0:
    # TODO: save checkpoint
    #     save_checkpoint()

    _logger.info('Epoch #%d validating' % epoch)
    valid_metric = evaluate_classification(model, val_loader, device, epoch, loss_func=loss_func,
                            steps_per_epoch=args.steps_per_epoch_val, tb_helper=tb)
    is_best_epoch = (
        valid_metric < best_valid_metric) if args.regression_mode else(
        valid_metric > best_valid_metric)
    if is_best_epoch:
        best_valid_metric = valid_metric
        if args.model_prefix and (args.backend is None or local_rank == 0):
            shutil.copy2(args.model_prefix + '_epoch-%d_state.pt' %
                         epoch, args.model_prefix + '_best_epoch_state.pt')
            # torch.save(model, args.model_prefix + '_best_epoch_full.pt')
    _logger.info('Epoch #%d: Current validation metric: %.5f (best: %.5f)' %
                 (epoch, valid_metric, best_valid_metric), color='bold')

'''

args = Args(
    data_train = datasets['train'],
    data_val = datasets['val'],
    data_test = datasets['test'],
    data_config = yaml_config,
    num_epochs = 5,
    batch_size = 64,
    model_prefix = 'models/torch_saved/student_models/',
    file_fraction = 1,
    data_fraction = 0.1,
    start_lr=1e-03,
    optimizer='ranger',
)

train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(args)

teacher_metadata = initialize_PELICAN(teacher_path, yaml_config, training=False, model_path = 'models/torch_saved/PELICAN_epoch-4_state.pt')

teacher_model = copy.deepcopy(teacher_metadata['model']).to(device)

metadata = initialize_models(network_path, yaml_config, training=True)
model = copy.deepcopy(metadata['model']).to(device)

loss_func = metadata['loss']
opt, scheduler = optim(args, model, device)

def kd_loop(kl_weight, temp, class_weight=1.0):

    model = copy.deepcopy(metadata['model']).to(device)
    opt, scheduler = optim(args, model, device)
    
    tb = None
    
    # training loop
    best_valid_metric = np.inf if args.regression_mode else 0
    grad_scaler = torch.amp.GradScaler("cuda")

    trial_name = 'Alpha = ' + str(kl_weight) + ', T = ' + str(temp)

    hyp_config = '_' + str(kl_weight).replace('.', '') + '_' + str(temp).replace('.', '')
    if class_weight == 0:
        print('KD Only!')
        hyp_config += '_kdonly'
    
    for epoch in range(args.num_epochs):
        if args.load_epoch is not None:
            if epoch <= args.load_epoch:
                continue
        _logger.info('-' * 50)
        _logger.info('Epoch #%d training' % epoch)
        knowledge_distillation(teacher_model, model, loss_func, opt, scheduler, train_loader, device, epoch, T=temp, kl_weight=kl_weight, class_weight=class_weight,
              steps_per_epoch=args.steps_per_epoch, grad_scaler=grad_scaler, tb_helper=tb)
        
        # if args.backend is not None and local_rank == 0:
        # TODO: save checkpoint
        #     save_checkpoint()

        hook_manager = {
    'forward_hooks': {
        'logits': 'fc'
    }}

        _logger.info('Epoch #%d validating' % epoch)
        valid_metric, valid_scores, valid_labels, valid_observers = evaluate_classification(model, val_loader, device, epoch, for_training=False, loss_func=loss_func,
                                steps_per_epoch=args.steps_per_epoch_val, tb_helper=tb)

        is_best_epoch = (
            valid_metric < best_valid_metric) if args.regression_mode else(
            valid_metric > best_valid_metric)
        if is_best_epoch:
            if args.model_prefix and (args.backend is None or local_rank == 0):
                dirname = os.path.dirname(args.model_prefix)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                state_dict = model.module.state_dict() if isinstance(
                    model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict()
                torch.save(state_dict, args.model_prefix + 'ResNet_student' + hyp_config + '_epoch-%d_state.pt' % epoch)
                torch.save(opt.state_dict(), args.model_prefix + 'ResNet_student' + hyp_config + '_epoch-%d_optimizer.pt' % epoch)
        
        _logger.info('Epoch #%d: Current validation metric: %.5f (best: %.5f)' %
                     (epoch, valid_metric, best_valid_metric), color='bold')

kl_weights = [1]
temps = [7]

for wt in kl_weights:
    for temp in temps:
        if wt == 0 and temp != 1:
            continue
        print(f'Starting! T = {temp}, Alpha = {wt}')
        kd_loop(wt, temp, class_weight=1.0)