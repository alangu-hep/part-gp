#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import glob
import argparse
workingdir = os.getenv("PATH_TO_PARTGP")
sys.path.append(os.path.abspath(workingdir))

import torch
import torch.nn as nn

from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.import_tools import import_module
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.nn.tools import evaluate_classification
from weaver.utils.logger import _logger, warn_n_times

from utils.nn_utils.part_prediction import test_load
from utils.nn_utils.hook_handler import register_forward_hooks, remove_all_forward_hooks, remove_handles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training = False
load_model = True
datasets_path = Path(os.getenv("PART_DATA"))
workingdir_path = Path(workingdir)

parser = argparse.ArgumentParser()

'''
Arguments
'''

# Optional Commands
parser.add_argument('--demo', action='store_true', default=False, 
                    help='Setup a demo that uses only 10% of the dataset')

# Network Loading
parser.add_argument('-m', '--target_modules', nargs = '*', default=['final_logits'], 
                    help='Target Modules for Hooking. SUPPORTED: ')
parser.add_argument('-n', '--network-script', type=str, default=str((workingdir_path / 'part_models/ParticleTransformer.py')), help='Network for ParT')

# Data
parser.add_argument('-d', '--data-type', type=str, default = 'full', help='Type of data for training')
parser.add_argument('-s', '--dataset', type=str, default = 'JetClass', help='The type of dataset used')
parser.add_argument('-c', '--data-config', type=str, default = str(workingdir_path / 'data_config/JetClass/JetClass_full.yaml'), 
                    help='Data Config YAML file')
parser.add_argument('-t', '--data-test', nargs = '*', default = glob.glob(str(datasets_path / 'JetClass/Pythia/train_100M') + '/*.root'), 
                    help='Dataset files for use')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--data-fraction', type=float, default=1.0,
                    help='Fraction of events to load from each file; for training, the events are randomly selected for each epoch')


# Extras

parser.add_argument('--extra-test-selection', type=str, default=None,
                    help='Additional test-time selection requirement, will modify `test_time_selection` to `(test_time_selection) & (extra)` on-the-fly')
parser.add_argument('--num-workers', type=int, default=1,
                    help='Number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with these numbers')


'''
Dictionaries
'''

weight_paths = {
    'full': 'part_models/ParT_full.pt',
    'kinpid': 'part_models/ParT_kinpid.pt',
    'kin': 'part_models/ParT_kin.pt'
}

jetclass_paths = {
    'train': 'JetClass/Pythia/train_100M',
    'validation': 'JetClass/Pythia/val_5M',
    'test': 'JetClass/Pythia/test_20M'
}

'''
Initialize Model
'''

args = parser.parse_args()

model_details = {}

network_module = import_module(args.network_script, name='_network_module')

data_config = SimpleIterDataset({}, args.data_config, for_training=training).config
module, model_info = network_module.get_model(data_config)

if load_model:
    weights_path = workingdir_path / weight_paths[args.data_type]
    wts = torch.load(str(weights_path), map_location = device, weights_only = True)
    module.load_state_dict(wts)

model_details['model'] = module
model_details['info'] = model_info
model_details['loss'] = network_module.get_loss(data_config)

model = model_details['model'].to(device)

print(f'\nSuccessfully Initialized {args.data_type} Model with the following modules:')

for name, module in model.named_modules():
    print(name)

interesting_layers = {
    'post_layer_embed': model.mod.embed.embed,
    'post_pair_embed': model.mod.pair_embed.embed,
    'first_layer_attn': model.mod.blocks[0].attn,
    'first_layer_block': model.mod.blocks[0],
    'final_layer_attn': model.mod.blocks[7].attn,
    'final_layer_block': model.mod.blocks[7],
    'first_cls_attn': model.mod.cls_blocks[0].attn,
    'first_cls_block': model.mod.cls_blocks[0],
    'final_cls_attn': model.mod.cls_blocks[1].attn,
    'final_cls_block': model.mod.cls_blocks[1],
    'final_logits': model.mod.fc
}

filtered_modules = {key: value for key, value in interesting_layers.items() if key in args.target_modules}

hook_outputs = {name: [] for name in filtered_modules.keys()}

remove_all_forward_hooks(model) # Safety Precaution

handles = register_forward_hooks(location_dict=filtered_modules, outputs=hook_outputs)

'''
Prediction
'''

test_loaders, data_config = test_load(args)

for name, get_test_loader in test_loaders.items():
    
    test_loader = get_test_loader()
    
    test_metric, scores, labels, observers = evaluate_classification(model, test_loader, device, epoch=None, for_training=False)
    _logger.info('Test metric %.5f' % test_metric, color='bold')
    del test_loader

remove_handles(handles)

print(f'Scores: \n{scores}')
print(f'Labels: \n{labels}')
print(f'Data: \n{observers}')