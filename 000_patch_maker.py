#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:52:24 2024

@author: ian
"""

import os
import skimage as ski
import numpy as np
from random import randint


def generate_random_patch(image, patch_size):
    labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
    x = randint(0, image.shape[1] - patch_size - 1)
    y = randint(0, image.shape[0] - patch_size - 1)
    if len(image.shape) == 3:
        ind = np.s_[y : y + patch_size, x : x + patch_size, :]
    elif len(image.shape) == 2:
        ind = np.s_[y : y + patch_size, x : x + patch_size]
    # (nrows, ncols, channels) = image.shape
    ind_label = np.s_[y : y + patch_size, x : x + patch_size]
    return image[ind], labels[ind_label], ind


def create_random_patch(im_path, patch_size=256):
    im = ski.io.imread(im_path)
    patch, labels, ind = generate_random_patch(im, patch_size)
    idx_string = (
        f"-x{ind[0].start}:{ind[0].stop}-y{ind[1].start}:{ind[1].stop}"
    )
    return patch, labels, idx_string


def save_patch(sample_name, patch_dir, idx_string):
    patch_path = os.path.join(patch_dir, sample_name + idx_string + ".tif")
    if not os.path.exists(patch_path):
        ski.io.imsave(patch_path, patch, check_contrast=False)


if __name__ == "__main__":
    raw_dir = "raw_images/"
    patch_dir = "test_patch/"
    n = 2
    patch_size = 512
    for im_name in os.listdir(raw_dir):
        im_path = os.path.join(raw_dir, im_name)
        sample_name = os.path.splitext(im_name)[0]
        for r in range(n):
            patch, labels, idx_string = create_random_patch(
                im_path, patch_size=patch_size
            )
            save_patch(sample_name, patch_dir, idx_string)
