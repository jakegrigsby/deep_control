import copy
import numbers
import os
import random
import subprocess
import time
from operator import itemgetter

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from skimage.util.shape import view_as_windows
from torch import nn
from torchvision import transforms


class AugmentationSequence:
    def __init__(self, aug_list):
        self.aug_list = aug_list

    def __call__(self, *batches):
        for aug in self.aug_list:
            aug.change_randomization_params()
        results = []
        for original_batch in batches:
            batch = original_batch.clone()
            for augmentation in self.aug_list:
                with torch.no_grad():
                    batch = augmentation(batch)
            results.append(batch)
        return tuple(results) if len(results) > 1 else results[0]


class GrayscaleAug:
    def __init__(self, batch_size, p_rand=0.5, *_args, **_kwargs):

        self.p_gray = p_rand
        self.batch_size = batch_size
        self.random_inds = np.random.choice(
            [True, False], batch_size, p=[self.p_gray, 1 - self.p_gray]
        )

    def grayscale(self, imgs):
        b, c, h, w = imgs.shape
        t = c // 3
        # TODO: speed this up
        for i in range(t):
            idx = i * 3
            imgs[:, idx : idx + 3] = (
                imgs[:, idx, ...] * 0.2989
                + imgs[:, idx + 1, ...] * 0.587
                + imgs[:, idx + 2, ...] * 0.114
            )
        return imgs

    def __call__(self, images):
        bs, channels, h, w = images.shape
        if self.random_inds.sum() > 0:
            images[self.random_inds] = self.grayscale(images[self.random_inds])
        return images

    def change_randomization_params(self):
        self.random_inds = np.random.choice(
            [True, False], self.batch_size, p=[self.p_gray, 1 - self.p_gray]
        )

    def print_parms(self):
        print(self.random_inds)


class CutoutAug:
    def __init__(
        self,
        batch_size,
        box_min=7,
        box_max=22,
        pivot_h=12,
        pivot_w=24,
        *_args,
        **_kwargs,
    ):

        self.box_min = box_min
        self.box_max = box_max
        self.pivot_h = pivot_h
        self.pivot_w = pivot_w
        self.batch_size = batch_size
        self.w1 = torch.randint(self.box_min, self.box_max, (batch_size,))
        self.h1 = torch.randint(self.box_min, self.box_max, (batch_size,))

    def __call__(self, imgs):
        n, c, h, w = imgs.shape
        for i, (img, w11, h11) in enumerate(zip(imgs, self.w1, self.h1)):
            img[
                :,
                self.pivot_h + h11 : self.pivot_h + h11 + h11,
                self.pivot_w + w11 : self.pivot_w + w11 + w11,
                ...,
            ] = 0
        return imgs

    def change_randomization_params(self, index_):
        self.w1[index_] = toch.randint(self.box_min, self.box_max)
        self.h1[index_] = torch.randint(self.box_min, self.box_max)

    def change_randomization_params(self):
        self.w1 = torch.randint(self.box_min, self.box_max, (self.batch_size,))
        self.h1 = torch.randint(self.box_min, self.box_max, (self.batch_size,))

    def print_parms(self):
        print(self.w1)
        print(self.h1)


class RadAug:
    """
    Emulates the augmentation in Reinforcement Learning with
    Augmented Data by upscaling and randomly cropping.
    """

    def __init__(self, batch_size, crop=16, *_args, **kwargs):
        self.batch_size = batch_size
        self.crop = crop
        self.change_randomization_params()

    def change_randomization_params(self):
        self.h = torch.randint(self.crop, size=(self.batch_size,))
        self.w = torch.randint(self.crop, size=(self.batch_size,))

    def resize_imgs(self, imgs):
        original_device = imgs.device
        b, c, h, w = imgs.shape
        result = np.empty((b, c, h + self.crop, w + self.crop), dtype=np.float32)
        for i in range(imgs.shape[0]):
            result[i] = cv2.resize(
                imgs[i].cpu().numpy().transpose(1, 2, 0), (h + self.crop, w + self.crop)
            ).transpose(2, 0, 1)
        return torch.from_numpy(result).to(original_device)

    def __call__(self, imgs):
        b, c, h, w = imgs.shape
        upscaled_imgs = self.resize_imgs(imgs)
        for i, (crop_h, crop_w) in enumerate(zip(self.h, self.w)):
            imgs[i, ...] = upscaled_imgs[i, :, crop_h : h + crop_h, crop_w : w + crop_w]
        return imgs


class DrqAug:
    """
    Emulates the augmentation in "Image Augmentation is All You Need"
    by replicating boundary pixels and then randomly cropping. We can't use
    the kornia library like the author's repo does, because there doesn't seem
    to be an easy way to preserve the crop across both observation batches. So the
    random crop is implemented manually. One thing I've noticed after looking at
    the official version is that kornia introduces many decimal pixel values during
    it's random crop. In case that's important, I've added a small amount of noise to
    get a similar effect.
    """

    def __init__(self, batch_size, pad=4, noise=True, *_args, **kwargs):
        self.batch_size = batch_size
        self.pad = pad
        self.change_randomization_params()
        self.noise = noise
        self.pad_func = nn.ReflectionPad2d(pad)

    def change_randomization_params(self):
        self.w1 = torch.randint(0, self.pad * 2, (self.batch_size,))
        self.h1 = torch.randint(0, self.pad * 2, (self.batch_size,))

    def random_crop(self, imgs, out):
        b, c, h, w = imgs.shape
        crop_max = h - out + 1
        cropped = torch.zeros((b, c, out, out), dtype=torch.float32)
        for i, (img, w11, h11) in enumerate(zip(imgs, self.w1, self.h1)):
            cropped[i] = img[:, h11 : h11 + out, w11 : w11 + out]
        return cropped

    def __call__(self, imgs):
        b, c, h, w = imgs.shape
        og_device = imgs.device
        padded = self.pad_func(imgs)
        cropped = self.random_crop(padded, out=h).to(og_device)
        if self.noise:
            cropped += torch.randn_like(cropped)
        return cropped.clamp(0, 255.0)

    def print_params(self):
        print(self.w1)
        print(self.h1)


class DrqNoNoiseAug(DrqAug):
    def __init__(self, batch_size, pad=4, noise=False, *_args, **kwargs):
        super().__init__(batch_size, pad, noise)


class LargeDrqNoNoiseAug(DrqAug):
    def __init__(self, batch_size, pad=12, noise=False, *_args, **kwargs):
        super().__init__(batch_size, pad, noise)


class LargeDrqAug(DrqAug):
    def __init__(self, batch_size, pad=12, *_args, **kwargs):
        super().__init__(batch_size, pad)


class TranslateAug:
    def __init__(
        self,
        batch_size,
        translate_max=4,
        *_args,
        **_kwargs,
    ):
        self.batch_size = batch_size
        self.translate_max = translate_max
        self.translation = torch.zeros((self.batch_size, 2), dtype=torch.float32)
        self.change_randomization_params()

    def change_randomization_params(self):
        self.translation = (
            torch.randint(
                2 * self.translate_max, (self.batch_size, 2), dtype=torch.int32
            )
            - self.translate_max
        )
        self.random_color = torch.randint(255, size=(self.batch_size, 3, 1, 1)).float()

    def __call__(self, imgs):
        b, c, h, w = imgs.shape
        final = torch.ones(
            (b, c, h + (2 * self.translate_max), w + (2 * self.translate_max)),
            dtype=torch.float32,
        )
        final *= self.random_color.repeat(1, c // 3, 1, 1)

        h_min = self.translate_max + self.translation[:, 0]
        h_max = self.translate_max + h + self.translation[:, 0]

        w_min = self.translate_max + self.translation[:, 1]
        w_max = self.translate_max + w + self.translation[:, 1]

        for i in range(self.batch_size):
            # TODO: speed this up??
            final[i, :, h_min[i] : h_max[i], w_min[i] : w_max[i]] = imgs[i]
        final = final[
            :,
            :,
            self.translate_max : -self.translate_max,
            self.translate_max : -self.translate_max,
        ].type(torch.float32)
        return final.to(imgs.device)


class LargeTranslateAug(TranslateAug):
    def __init__(self, batch_size, translate_max=8, *_args, **kwargs):
        super().__init__(batch_size, translate_max)


class CutoutColorAug:
    def __init__(
        self,
        batch_size,
        box_min=7,
        box_max=22,
        pivot_h=12,
        pivot_w=24,
        *_args,
        **_kwargs,
    ):
        self.box_min = box_min
        self.box_max = box_max
        self.pivot_h = pivot_h
        self.pivot_w = pivot_w
        self.batch_size = batch_size
        self.change_randomization_params()

    def __call__(self, imgs):
        b, c, h, w = imgs.shape
        self.rand_box = self.rand_box.to(imgs.device)
        for i, (img, w11, h11) in enumerate(zip(imgs, self.w1, self.h1)):
            for t in range(c // 3):
                img[
                    t * 3 : (t + 1) * 3,
                    self.pivot_h + h11 : self.pivot_h + h11 + h11,
                    self.pivot_w + w11 : self.pivot_w + w11 + w11,
                ] = self.rand_box[i]
        return imgs

    def change_randomization_params(self):
        self.w1 = torch.randint(self.box_min, self.box_max, (self.batch_size,))
        self.h1 = torch.randint(self.box_min, self.box_max, (self.batch_size,))
        self.rand_box = torch.randint(
            0,
            255,
            size=(self.batch_size, 3, 1, 1),
            dtype=torch.float32,
        )

    def print_parms(self):
        print(self.w1)
        print(self.h1)


class GammaAug:
    gamma_mean = 1.0
    gamma_std = 0.45

    def __init__(self, batch_size, *_args, **kwargs):
        self.batch_size = batch_size
        self.change_randomization_params()

    def change_randomization_params(self):
        self.gamma = torch.from_numpy(
            np.random.normal(self.gamma_mean, self.gamma_std, size=(self.batch_size,))
        ).float()
        self.gamma = self.gamma.view(-1, 1, 1, 1)

    def __call__(self, imgs):
        imgs /= 255.0
        imgs = imgs.pow(self.gamma.to(imgs.device))
        imgs *= 255.0
        return imgs.clamp(0.0, 255.0)


class _FlipAug:
    def __init__(self, batch_size, p_rand=0.5, dim=None, *_args, **_kwargs):
        assert dim
        self.p_flip = p_rand
        self.batch_size = batch_size
        self.dim = dim
        self.change_randomization_params()

    def __call__(self, images):
        if self.random_inds.sum() > 0:
            images[self.random_inds] = torch.flip(images[self.random_inds], (self.dim,))
        return images

    def change_randomization_params(self):
        self.random_inds = np.random.choice(
            [True, False], self.batch_size, p=[self.p_flip, 1 - self.p_flip]
        )

    def print_parms(self):
        print(self.random_inds)


class HorizontalFlipAug(_FlipAug):
    def __init__(self, batch_size, p_rand=0.5, *_args, **kwargs):
        super().__init__(batch_size, p_rand, dim=3)


class VerticalFlipAug(_FlipAug):
    def __init__(self, batch_size, p_rand=0.5, *_args, **kwargs):
        super().__init__(batch_size, p_rand, dim=2)


class RotateAug:
    def __init__(self, batch_size, *_args, **_kwargs):

        self.batch_size = batch_size
        self.change_randomization_params()

    def __call__(self, imgs):
        for k in range(1, 4):
            rotate_mask = torch.where((self.random_inds == k))
            imgs[rotate_mask] = torch.rot90(imgs[rotate_mask], k=(k + 1), dims=(2, 3))
        return imgs

    def change_randomization_params(self):
        self.random_inds = (
            torch.randint(
                4,
                size=(self.batch_size,),
            )
            * self.batch_size
            + np.arange(self.batch_size)
        )

    def print_parms(self):
        print(self.random_inds)


class IdentityAug:
    def __init__(self, batch_size, *_args, **_kwargs):
        self.batch_size = batch_size

    def __call__(self, imgs):
        return imgs

    def change_randomization_params(self):
        return

    def print_parms(self):
        return


class WindowAug:
    def __init__(self, batch_size, *_args, **_kwargs):
        self.batch_size = batch_size
        self.crop_size = 64
        self.crop_max = 75 - self.crop_size
        self.change_randomization_params()

    def __call__(self, imgs):
        mask = torch.zeros((imgs.shape), dtype=torch.float32, device=imgs.device)
        for i in range(self.batch_size):
            mask[
                i,
                :,
                self.h1[i] : self.h1[i] + self.crop_size,
                self.w1[i] : self.w1[i] + self.crop_size,
            ] = 1.0
        imgs *= mask
        return imgs

    def change_randomization_params(self):
        self.w1 = torch.randint(0, self.crop_max, (self.batch_size,))
        self.h1 = torch.randint(0, self.crop_max, (self.batch_size,))

    def print_parms(self):
        print(self.w1)
        print(self.h1)


class ColorJitterAug(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.5,
        p_rand=1.0,
        stack_size=1,
        *_args,
        **_kwargs,
    ):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )

        self.prob = p_rand
        self.batch_size = batch_size
        self.stack_size = stack_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.change_randomization_params()

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        means = torch.mean(x, dim=(2, 3), keepdim=True)
        return torch.clamp(
            (x - means) * self.factor_contrast.view(len(x), 1, 1, 1) + means, 0, 1
        )

    def adjust_hue(self, x):
        h = x[:, 0, :, :]
        h += self.factor_hue.view(len(x), 1, 1) * 255.0 / 360.0
        h = h % 1
        x[:, 0, :, :] = h
        return x

    def adjust_brightness(self, x):
        x[:, 2, :, :] = torch.clamp(
            x[:, 2, :, :] * self.factor_brightness.view(len(x), 1, 1), 0, 1
        )
        return torch.clamp(x, 0, 1)

    def adjust_saturate(self, x):
        x[:, 1, :, :] = torch.clamp(
            x[:, 1, :, :] * self.factor_saturate.view(len(x), 1, 1), 0, 1
        )
        return torch.clamp(x, 0, 1)

    def transform(self, inputs):
        hsv_transform_list = [
            rgb2hsv,
            self.adjust_brightness,
            self.adjust_hue,
            self.adjust_saturate,
            hsv2rgb,
        ]
        rgb_transform_list = [self.adjust_contrast]

        # Shuffle transform
        if random.uniform(0, 1) >= 0.5:
            transform_list = rgb_transform_list + hsv_transform_list
        else:
            transform_list = hsv_transform_list + rgb_transform_list
        for t in transform_list:
            inputs = t(inputs)
        return inputs

    def __call__(self, imgs):
        b, c, h, w = imgs.shape
        original_device = imgs.device
        imgs = imgs.float().to(self._device)
        imgs = imgs / 255.0

        t = c // 3
        for i in range(t):
            idx = i * 3
            imgs[:, idx : idx + 3] = self.forward(imgs[:, idx : idx + 3])
        imgs = imgs.to(original_device) * 255.0
        return imgs

    def change_randomization_params(self):
        factor_contrast = torch.empty(self.batch_size, device=self._device).uniform_(
            *self.contrast
        )
        self.factor_contrast = (
            factor_contrast.reshape(-1, 1).repeat(1, self.stack_size).reshape(-1)
        )

        factor_hue = torch.empty(self.batch_size, device=self._device).uniform_(
            *self.hue
        )
        self.factor_hue = (
            factor_hue.reshape(-1, 1).repeat(1, self.stack_size).reshape(-1)
        )

        factor_brightness = torch.empty(self.batch_size, device=self._device).uniform_(
            *self.brightness
        )
        self.factor_brightness = (
            factor_brightness.reshape(-1, 1).repeat(1, self.stack_size).reshape(-1)
        )

        factor_saturate = torch.empty(self.batch_size, device=self._device).uniform_(
            *self.saturation
        )
        self.factor_saturate = (
            factor_saturate.reshape(-1, 1).repeat(1, self.stack_size).reshape(-1)
        )

    def print_parms(self):
        print(self.factor_hue)

    def forward(self, inputs):
        random_inds = np.random.choice(
            [True, False], len(inputs), p=[self.prob, 1 - self.prob]
        )
        inds = torch.tensor(random_inds).to(self._device)
        if random_inds.sum() > 0:
            inputs[inds] = self.transform(inputs[inds])
        return inputs


def rgb2hsv(rgb, eps=1e-8):
    # Reference: https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = rgb.device
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin

    hue = torch.zeros((rgb.shape[0], rgb.shape[2], rgb.shape[3])).to(_device)
    hue[Cmax == r] = (((g - b) / (delta + eps)) % 6)[Cmax == r]
    hue[Cmax == g] = ((b - r) / (delta + eps) + 2)[Cmax == g]
    hue[Cmax == b] = ((r - g) / (delta + eps) + 4)[Cmax == b]
    hue[Cmax == 0] = 0.0
    hue = hue / 6.0  # making hue range as [0, 1.0)
    hue = hue.unsqueeze(dim=1)

    saturation = (delta) / (Cmax + eps)
    saturation[Cmax == 0.0] = 0.0
    saturation = saturation.to(_device)
    saturation = saturation.unsqueeze(dim=1)

    value = Cmax
    value = value.to(_device)
    value = value.unsqueeze(dim=1)

    return torch.cat(
        (hue, saturation, value), dim=1
    )  # .type(torch.FloatTensor).to(_device)
    # return hue, saturation, value


def hsv2rgb(hsv):
    # Reference: https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = hsv.device

    hsv = torch.clamp(hsv, 0, 1)
    hue = hsv[:, 0, :, :] * 360.0
    saturation = hsv[:, 1, :, :]
    value = hsv[:, 2, :, :]

    c = value * saturation
    x = -c * (torch.abs((hue / 60.0) % 2 - 1) - 1)
    m = (value - c).unsqueeze(dim=1)

    rgb_prime = torch.zeros_like(hsv).to(_device)

    inds = (hue < 60) * (hue >= 0)
    rgb_prime[:, 0, :, :][inds] = c[inds]
    rgb_prime[:, 1, :, :][inds] = x[inds]

    inds = (hue < 120) * (hue >= 60)
    rgb_prime[:, 0, :, :][inds] = x[inds]
    rgb_prime[:, 1, :, :][inds] = c[inds]

    inds = (hue < 180) * (hue >= 120)
    rgb_prime[:, 1, :, :][inds] = c[inds]
    rgb_prime[:, 2, :, :][inds] = x[inds]

    inds = (hue < 240) * (hue >= 180)
    rgb_prime[:, 1, :, :][inds] = x[inds]
    rgb_prime[:, 2, :, :][inds] = c[inds]

    inds = (hue < 300) * (hue >= 240)
    rgb_prime[:, 2, :, :][inds] = c[inds]
    rgb_prime[:, 0, :, :][inds] = x[inds]

    inds = (hue < 360) * (hue >= 300)
    rgb_prime[:, 2, :, :][inds] = x[inds]
    rgb_prime[:, 0, :, :][inds] = c[inds]

    rgb = rgb_prime + torch.cat((m, m, m), dim=1)
    rgb = rgb.to(_device)

    return torch.clamp(rgb, 0, 1)


class NetworkRandomizationAug(torch.nn.Module):
    def __init__(self, batch_size, *_args, **_kwargs):
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.change_randomization_params()

    def forward(self, x):
        return self.conv(x)

    def __call__(self, imgs):
        b, c, h, w = imgs.shape
        t = c // 3
        imgs /= 255.0
        for i in range(t):
            idx = i * 3
            with torch.no_grad():
                imgs[:, idx : idx + 3] = self.forward(imgs[:, idx : idx + 3])
        imgs *= 255.0
        return imgs

    def change_randomization_params(self):
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(
            self._device
        )
        torch.nn.init.xavier_normal_(self.conv.weight.data)

    def print_parms(self):
        print(self.conv.weight.data)
