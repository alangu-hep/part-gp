#!/usr/bin/env python3

import pysr
from pysr import PySRRegressor

import sys
import os

from pathlib import Path
import glob

import numpy as np
import torch
import torch.nn as nn
import copy
import shutil

from weaver.utils.import_tools import import_module
from src.preprocessing.datasets import SimpleIterDataset
from weaver.utils.nn.tools import evaluate_classification, train_classification
from weaver.utils.logger import _logger, warn_n_times

from src.part_prediction import test_load, train_load, knowledge_distillation, optim, vae_testing, train_autoencoder, vae_diagnostic, prepare_logits
from src.handlers.hook_handler import HookHandler

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
        self.lr_scheduler = 'flat+decay'
        self.load_epoch = None
        self.gpus = 0
        self.predict_gpus = 0
        self.regression_mode = False
        
        for key, value in kwargs.items():
            setattr(self, key, value)

yaml_config = 'data_config/JetClass/JetClass_kin.yaml'
network_path = 'models/networks/part_wrapper.py'
encoder_path = 'models/networks/autoencoder.py'

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

complexity = {
    'particle_attn': 2,
    'class_attn': 1
}

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

def initialize_VAE(network_path, config_path, training = True, model_path = None) -> dict:

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
Generating the SR Dataset
'''

part_metadata = initialize_models(network_path, yaml_config, training=False, model_path = 'models/torch_saved/student_models/ParT_A1_T7_epoch-4_state.pt')
teacher = copy.deepcopy(part_metadata['model']).to(device)

encoder_metadata = initialize_VAE(encoder_path, yaml_config, training=False, model_path = 'models/torch_saved/DSVAE_epoch-4_state.pt')
model = copy.deepcopy(encoder_metadata['model']).to(device)

#loss_func = torch.nn.MSELoss()

pred_args = Args(
    data_test = datasets['train'],
    data_config = yaml_config,
    batch_size = 64,
    file_fraction = 1,
    data_fraction = 0.01
)

test_loaders, data_config = test_load(pred_args)

train_size = (1e+08 * (num_classes / 10)) * pred_args.data_fraction
val_size = (5e+06 * (num_classes / 10)) * pred_args.data_fraction
test_size = (2e+07 * (num_classes / 10)) * pred_args.data_fraction

for name, get_test_loader in test_loaders.items():
    
    test_loader = get_test_loader()
    
    latents, logits = prepare_logits(model, teacher, test_loader, device, epoch=None, dataset_size=train_size, for_training=False)
    
    del test_loader

logits = torch.cat(logits).detach().cpu().numpy()
inputs = torch.cat(latents).detach().cpu().numpy()

outputdir = 'outputs/pysr_outputs/sr_tests'

model = PySRRegressor(
    maxsize=40,
    niterations=5600,
    populations=48,
    population_size = 27,
    ncycles_per_iteration = 1520,
    weight_optimize=0.001,
    binary_operators=[
        "+", 
        "-", 
        "*", 
        "/", 
        "^",
    ],
    unary_operators = [
        "sqrt", 
        "tanh",
        "sin",
    ],
    constraints = {
        '^': (-1, 1)
    },
    nested_constraints = {
        "*": {"tanh": 2},
        "tanh": {"tanh": 0, "^": 1, "sin": 1},
        "sin": {"sin": 0}       
    }, 
    output_directory = outputdir,
    parsimony = 0.01,
    annealing=True,
    batching=True,
    elementwise_loss = 'HuberLoss(1.35)',
    random_state=42
)

model.fit(inputs, logits)

