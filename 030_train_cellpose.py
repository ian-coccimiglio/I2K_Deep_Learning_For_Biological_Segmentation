#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:28:32 2024

@author: ian
"""

import os
from cellpose import models, io, train
import logging

logging.basicConfig(level=logging.INFO)

patch_dir = "patch"
label_dir = "patch_label"
model_path = "models"
patches = []
labels = []
print("Loading patches and labels for training")


def read_matching_images(patch_name, label_name):
    patch_path = os.path.join(patch_dir, patch_name)
    label_path = os.path.join(label_dir, label_name)
    patch = io.imread(patch_path)
    label = io.imread(label_path)
    return patch, label


for label_name in os.listdir(label_dir):
    patch_name = label_name.split("_label")[0] + ".tif"
    patch, label = read_matching_images(patch_name, label_name)
    print(f"Reading {patch_name} matched to {label_name}")
    patches.append(patch)
    labels.append(label)

model = models.CellposeModel(gpu=True, model_type="cyto3")

n = 800
model_path = train.train_seg(
    model.net,
    train_data=patches,
    train_labels=labels,
    channels=[0, 0],
    n_epochs=n,
    model_name=f"psr_{n}epochs_{len(labels)}patches",
)
