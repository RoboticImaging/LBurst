# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

# Modified for RoBLo feature extraction

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob

from nets.patchnet import *
from tools import common
from tools.cstack_dataloader import norm_RGB, \
    tensor2img
from tools.burst_generation import generate_singleburst, center_crop, add_noise
from tools.burst_dataloader import norm as norm_burst


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + str(checkpoint['net']))
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(
        net)
    print(
        f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")

    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(
            self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1,
                                             padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        maxima = (repeatability == self.max_filter(repeatability))
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale(net, img, detector, scale_f=2 ** 0.25,
                       min_scale=0.0, max_scale=1,
                       min_size=256, max_size=1024,
                       burst_size=1,
                       verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    print(img.shape)

    # Extract RoBLo features
    B, channels, H, W = img.shape
    assert B == 1 and channels == 3*burst_size, "should be a batch with a single RGB image"
    assert max_scale <= 1
    s = 1.0

    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")

            with torch.no_grad():
                res = net(imgs=[img])

            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            y, x = detector(
                **res)
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)
    scores = torch.cat(C) * torch.cat(Q)
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


def extract_keypoints(args):
    iscuda = common.torch_set_gpu(args.gpu)

    net = load_network(args.model)
    if iscuda: net = net.cuda()

    detector = NonMaxSuppression(
        rel_thr=args.reliability_thr,
        rep_thr=args.repeatability_thr)

    while args.images:
        img_path = args.images.pop(0)

        if img_path.endswith('.txt'):
            args.images = open(
                img_path).read().splitlines() + args.images

        if os.path.isdir(img_path):
            img_path = glob.glob(f"{img_path}/*png")
            img_path.sort()

            burst = []
            print(f"\nExtracting features for {img_path}")

            for file in img_path:
                image = Image.open(file).convert('RGB')
                burst.append(np.array(image))
            img = np.concatenate(burst, axis=2)
            W, H = image.size
            img = norm_burst(img)[None]
            img_path = img_path[0]
        else:
            print(f"\nExtracting features for {img_path}")
            orig_img = Image.open(img_path).convert('RGB')
            W, H = orig_img.size
            img, _ = get_input_img(args, orig_img)

        if iscuda: img = img.cuda()

        # repeat RoBLo feature extraction over multiple scales
        xys, desc, scores = extract_multiscale(net, img, detector,
                                               scale_f=args.scale_f,
                                               burst_size=args.burst_size,
                                               min_scale=args.min_scale,
                                               max_scale=args.max_scale,
                                               min_size=args.min_size,
                                               max_size=args.max_size,
                                               verbose=True)

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[
               -args.top_k or None:]

        outpath = img_path + '.' + args.tag
        print(f"Saving {len(idxs)} keypoints to {outpath}")
        np.savez(open(outpath, 'wb'),
                 imsize=(W, H),
                 keypoints=xys[idxs],
                 descriptors=desc[idxs],
                 scores=scores[idxs])


def get_input_img(args, orig_img):
    burst_buffer = 30
    adjusted_crop_size = args.crop_size - 2*burst_buffer
    output_img = np.array(orig_img)

    if args.crop_size > 0:
        output_img = center_crop(output_img, [args.crop_size, args.crop_size])

    if args.burst_size > 1:
        burst = generate_singleburst(orig_img, burst_size=args.burst_size, crop_size=adjusted_crop_size,
                                     variance=args.noise_var)
        centre_start = args.burst_size // 2 * 3
        output_img = burst[:, :, centre_start:centre_start + 3]
        img = norm_burst(burst)[None]
    else:
        output_img = center_crop(output_img, [adjusted_crop_size, adjusted_crop_size])
        if args.noise_var > 0:
            mean = 0
            output_img = np.stack(
                ([add_noise(output_img[:, :, k], mean, args.noise_var) for k in range(output_img.shape[2])]), axis=2)
        img = norm_burst(output_img)[None]
    return img, output_img



# command-line arguements for RoBLo features.
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Extract keypoints for a given image")

    parser.add_argument("--model", type=str, default='models/RoBLo_N16_B5.pt', help='model path')
    parser.add_argument("--images", type=str, required=True, nargs='+', help='images / list')

    #If you have a single image to generate a single robotic burst, uncomment the following and change img_path = args.images.pop(0) to args.images
    #parser.add_argument("--images", type=str, default="['imgs/toyimg2.png']", nargs='+',help='images / list')  # define image path, or multiple images as list

    parser.add_argument("--tag", type=str, default='RoBLo', help='output file tag')
    parser.add_argument("--top-k", type=int, default=5000, help='number of keypoints') # Number of maximum RoBLo features
    parser.add_argument("--burst-raw", type=int, default=1)   # input is a burst structure - 1; for synthetic burst generation 0
    parser.add_argument("--burst-size", type=int, default=1)  # single image approach - 1, burst of images - default 5
    parser.add_argument("--crop-size", type=int, default=0)   # size of centre crop, for no crop, set 0
    parser.add_argument("--noise-var", type=int, default=0)   # noise variance on a scale of 0 - 255, set 0 for no additional noise
    parser.add_argument("--scale-f", type=float, default=2 ** 0.25)  # scale is defined as 2^(1/4) similar to SIFT and R2D2
    parser.add_argument("--min-size", type=int, default=256)  # minimum size of an image 256
    parser.add_argument("--max-size", type=int, default=1024) # maximum size of an image 1024
    parser.add_argument("--min-scale", type=float, default=0) # increasing scale demonstrates improvement at the expense of computation
    parser.add_argument("--max-scale", type=float, default=1) # original scale of an image

    parser.add_argument("--reliability-thr", type=float, default=0.7)   # threshold for reliability, default 0.7
    parser.add_argument("--repeatability-thr", type=float, default=0.7) # threshold for repeatability, default 0.7

    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='use -1 for CPU')
    args = parser.parse_args()

    extract_keypoints(args)
