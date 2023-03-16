# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

# Modified for RoBLo feature extraction

import torch
from nets.patchnet import *
from tools import common


def load_network(model_fn, net=None, ignore_keys=[]):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net'])
    if net is None:
        net = eval(checkpoint['net'])
    else:
        saved_net = eval(checkpoint['net'])
        assert isinstance(net,
                          saved_net.__class__), f"The loaded model args.model {saved_net.__class__} needs to be the same class as the new network, args.net {net.__class__}"

    nb_of_weights = common.model_size(net)
    print(
        f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")

    weights = checkpoint['state_dict']
    for k in ignore_keys:
        del weights[k]
    ret = net.load_state_dict(weights, strict=False)
    print(f"No of missing keys: {len(ret.missing_keys)}")
    return net.eval()
