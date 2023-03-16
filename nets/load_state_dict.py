# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

# Modified for RoBLo feature extraction

import torch

def load_state_dict(model_fn):
    checkpoint = torch.load(model_fn)
    weights = checkpoint['state_dict']
    return {k.replace('module.', ''): v for k, v in weights.items()}
