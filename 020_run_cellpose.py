#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:14:24 2024

@author: ian
"""

from cellpose import models, io
import numpy as np
import os
import napari
import warnings


def napari_multiview(im, label):
    """
    View an image and its label in napari.
    """
    viewer = napari.Viewer()
    viewer.add_image(im)
    viewer.add_labels(label)
    napari.run()


def run_cellpose_model(useGPU=True, model=None):
    if (model is not None) and (model != "cyto3"):
        model = models.CellposeModel(
            gpu=useGPU,
            pretrained_model=model_path,
        )
        masks, flow, style = model.eval(im, diameter=None, channels=[0, 0])
    else:
        model = models.Cellpose(gpu=useGPU, model_type="cyto3")
        masks, flow, style, diam = model.eval(
            im, diameter=None, channels=[0, 0]
        )
        print(f"estimated diameter: {np.round(diam,1)}")

    return masks


# warnings.filterwarnings("ignore")

base_model_name = "cyto3"
custom_model_dir = "models"
model_name = "psr_400epochs_3patches"

n = 0
image_dir = "raw_images"
image_names = sorted(os.listdir(image_dir))
image_paths = [
    os.path.join(image_dir, image_name) for image_name in image_names
]

useGPU = True
view_napari = True

if __name__ == "__main__":
    model_path = os.path.join(custom_model_dir, model_name)

    for enum, image_path in enumerate(image_paths):
        print(f"Running {model_name} model on: {image_path}")
        im = io.imread(image_path)
        masks = run_cellpose_model(useGPU=useGPU, model=model_path)

        if view_napari:
            napari_multiview(im, masks)
