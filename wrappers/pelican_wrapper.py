# TODO

import torch
from .PELICAN.models.pelican_classifier import PELICANClassifier
from weaver.utils.logger import _logger

class PELICANWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = PELICANClassifier(**kwargs)
        self.dataset = kwargs.get('dataset', '')
        self.scale_factor = kwargs.get('scale', 1.0)

    def forward(self, points, features, lorentz_vectors, mask):
        
        reordered = torch.stack([
            lorentz_vectors[:, 3, :],
            lorentz_vectors[:, 0, :],
            lorentz_vectors[:, 1, :],
            lorentz_vectors[:, 2, :]
        ], dim=1) * self.scale_factor

        mask_bool = torch.flatten(mask.bool(), start_dim=1)
        
        data = {
            'Pmu': torch.transpose(reordered, -2, -1),
            'particle_mask': mask_bool,
            'Nobj': mask.sum(dim=(1, 2)),
            'edge_mask': mask_bool.unsqueeze(1) * mask_bool.unsqueeze(2)
        }
        
        if self.dataset == 'jc':
            features_transposed = features.transpose(-2, -1)
            data['part_charge'] = features_transposed[:, :, 5].long()
            data['part_isChargedHadron'] = features_transposed[:, :, 6]
            data['part_isNeutralHadron'] = features_transposed[:, :, 7]
            data['part_isPhoton'] = features_transposed[:, :, 8]
            data['part_isElectron'] = features_transposed[:, :, 9]
            data['part_isMuon'] = features_transposed[:, :, 10]
            data['d0val'] = features_transposed[:, :, 11] 
            data['d0err'] = features_transposed[:, :, 12]
            data['dzval'] = features_transposed[:, :, 13]
            data['dzerr'] = features_transposed[:, :, 14]
        
        return self.mod(data)['predict']


def get_model(data_config, **kwargs):

    cfg = dict(
        rank1_dim_multiplier=1,
        num_channels_scalar=10,
        num_channels_m=[[60],]*5,
        num_channels_2to2=[35,]*5,
        num_channels_out=[60],
        num_channels_m_out=[60, 35],
        stabilizer='so2',
        method='spurions',
        num_classes=len(data_config.label_value),
        average_nobj=32,
        drop_rate=0.01, 
        drop_rate_out=0.01,
        dataset='',
        config='M',
        config_out='M',
        scale=0.1
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = PELICANWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info
