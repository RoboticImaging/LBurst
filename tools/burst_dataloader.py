# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

# Modified for RoBLo feature extraction

import numpy as np
import torch
import torchvision.transforms as tvf
from tools.transforms import instanciate_transformation

from tools.burst_generation import generate_burst, generate_singleburst, center_crop

RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])


class BurstTrainingLoader:
    """ On-the-fly loader of central image and ground truth and corresponding noisy burst.
    
    self[idx] returns a dictionary with keys: img1, img1_X, img1_R, img1_S, burst
     - img1: cropped original
     - img1_X: descriptors of img1
     - img1_S: repeatability map of img1
     - img1_R: reliability map of img1
     - burst based on image from idx with img1 as central image

    """

    def __init__(self, dataset, dim=128, burst_size=5, crop_size=500, noise_var=10, idx_as_rng_seed=False, crop=''):
        self.burst_size = burst_size
        self.crop_size = crop_size
        self.noise_var = noise_var
        self.dataset = dataset
        self.dim = dim
        self.idx_as_rng_seed = idx_as_rng_seed # to remove randomness

        self.crop = instanciate_transformation(crop)

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = 'BurstTrainingLoader\n'
        fmt_str += repr(self.dataset)
        fmt_str += '  npairs: %d\n' % self.dataset.npairs
        short_repr = lambda s: repr(s).strip().replace('\n', ', ')[14:-1].replace('    ', ' ')
        return fmt_str

    def __getitem__(self, i):
        # from time import time as now; t0 = now()
        if self.idx_as_rng_seed:
            import random
            random.seed(i)
            np.random.seed(i)

        # Retrieve an image pair and their absolute flow
        img_orig = self.dataset.get_image(i)
        burst_buffer = 30
        if img_orig.size[0] < self.crop_size or img_orig.size[1] < self.crop_size:
            print(f"Resizing from {img_orig.size} to {(self.crop_size, self.crop_size)}")
            img_orig = img_orig.resize((self.crop_size, self.crop_size))

        img_a = img_orig
        img_a = np.array(img_a)
        ah, aw, p1 = img_a.shape
        assert p1 == 3

        burst = generate_singleburst(img_a, crop_size=self.crop_size-burst_buffer*2, burst_size=self.burst_size, variance=self.noise_var)
        burst = np.uint8(burst * 255)

        centre_start = self.burst_size // 2 * 3
        img1 = burst[:, :, centre_start:centre_start+3]
        result = dict(img1=norm(img1), burst=[norm(burst)])
        return result

def norm(img):
    if img.shape[0] == 3:
        return norm_RGB(img)
    else:
        imgs = np.split(img, int(img.shape[2] / 3), axis=2)
        norm_imgs = [norm_RGB(i) for i in imgs]
        norm_imgs = torch.cat(norm_imgs, dim=0)

        return norm_imgs
