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


def run_cellpose_model(
    useGPU=True, model_name=None, custom_model_dir="models"
):
    if (model_name is not None) and (model_name != "cyto3"):
        model_path = os.path.join(custom_model_dir, model_name)
        model = models.CellposeModel(
            gpu=useGPU,
            pretrained_model=model_path,
        )
    else:
        model = models.CellposeModel(gpu=useGPU, model_type="cyto3")

    masks, flow, style = model.eval(
        im,
        diameter=None,
        channels=[0, 0],
        cellprob_threshold=0.0,
        flow_threshold=0.4,
    )

    return masks


# warnings.filterwarnings("ignore")
custom_model_dir = "models"
model_name = "psr_800epochs_6patches"
image_dir = "raw_images"


n = 0
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
        masks = run_cellpose_model(
            useGPU=useGPU,
            model_name=model_name,
            custom_model_dir=custom_model_dir,
        )

        if view_napari:
            napari_multiview(im, masks)
