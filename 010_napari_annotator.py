#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:30:29 2024

@author: ian
"""

import napari
import os
import numpy as np
import skimage as ski


def load_patch(patch_path):
    if os.path.exists(patch_path):
        patch = ski.io.imread(patch_path)
        return patch
    else:
        return None


def parse_name(filepath):
    fn = os.path.split(filepath)[-1]
    sample_name = fn.split("-x")[0]
    xcoord = fn.split("-x")[1].split("-")[0].split(":")
    ycoord = fn.split("-y")[1].split(".")[0].split(":")
    xcoords = [int(x) for x in xcoord]
    ycoords = [int(y) for y in ycoord]
    square = np.array(
        [
            np.array([xcoords[0], ycoords[0]]),
            np.array([xcoords[0], ycoords[1]]),
            np.array([xcoords[1], ycoords[1]]),
            np.array([xcoords[1], ycoords[0]]),
        ]
    )
    return sample_name, square


def get_square(patch_path, patch=None):
    sample_name, square = parse_name(patch_path)
    raw_dir = "raw_images"
    im_path = os.path.join(raw_dir, sample_name + ".jpeg")
    im = ski.io.imread(im_path)
    return (im, square)


def show_squares(im, squares, patches=None):
    viewer = napari.Viewer()
    viewer.add_image(im)

    viewer.add_shapes(
        squares,
        face_color="blue",
        edge_color="green",
        name="bounding box",
        edge_width=3,
    )

    if patches is not None:
        for patch in patches:
            viewer.add_image(patch, translate=[square[0][0], square[0][1]])

    napari.run()


def annotate_patch(patch_path, patch_label_path, save_path=None):
    patch = load_patch(patch_path)
    label = load_patch(patch_label_path)
    if label is None:
        label = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint16)

    if patch is not None:
        viewer = napari.Viewer()
        patch_layer = viewer.add_image(patch)
        label_layer = viewer.add_labels(label)
        label_layer.brush_size = 4
        label_layer.mode = "paint"
        label_layer.preserve_labels = True
        napari.run()

    if save_path is None:
        ski.io.imsave(patch_label_path, label_layer.data, check_contrast=False)
    else:
        ski.io.imsave(save_path, label_layer.data, check_contrast=False)


if __name__ == "__main__":
    patch_dir = "patch/"
    patch_label_dir = "patch_label/"
    sample = "noisyPSR"  # CHANGE THIS
    annotate = True  # CHANGE THIS

    squares = []
    for patch_name in os.listdir(patch_dir):
        if patch_name.startswith(sample):
            patch_path = os.path.join(patch_dir, patch_name)
            label_dir = os.path.join(patch_label_dir)
            patch_label_path = (
                os.path.splitext(os.path.join(label_dir, patch_name))[0]
                + "_label.png"
            )
            patch = load_patch(patch_path)
            im, square = get_square(patch_path)
            squares.append(square)
            if annotate:
                os.makedirs(patch_label_dir, exist_ok=True)
                annotate_patch(patch_path, patch_label_path)
    # For quality control:
    if not annotate:
        show_squares(im, squares)
