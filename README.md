# I2K\_Deep\_Learning\_For\_Biological\_Segmentation
Developing a Napari-based Workflow for Biological Instance Segmentation.

## Interactive vs. Observing 
For the interactive version of this workshop, you need to have a working knowledge of the terminal and Python.
To listen in on the workshop, you simply need to be curious about connection between deep-learning and image segmentation. 

## Workshop Details:
Today, we'll be talking about how to perform semi-supervised finetuning for instance segmentation. In particular, we'll be taking advantage of napari's builtin functionality for quickly creating labels and annotating images.

## (Interactive) Installation
Installation guidelines require a version of conda. ![Miniconda]((https://docs.anaconda.com/miniconda/).
If you're planning on attending Brian Northan's workshop, my workshop is good prerequisite to his. You can reuse the environment and simply install the packages you're missing after.

### Linux

```python
conda create -n I2K2024_dl python=3.10
conda activate I2K2024_dl
pip install numpy==1.26 
pip install "napari[all]" # requires quotation marks to parse the square brackets on Linux, not sure if generalizes
pip install albumentations
pip install matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cellpose
```

### Windows

```python
conda create -n I2K2024_dl python=3.10
conda activate I2K2024_dl
pip install numpy==1.26 
pip install napari[all]
pip install albumentations
pip install matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cellpose
```

### Mac

```python
conda create -n I2K2024_dl python=3.10
conda activate I2K2024_dl
pip install numpy==1.26 
pip install "napari[all]" # requires quotation marks to parse the square brackets on Linux, not sure if generalizes
pip install albumentations
pip install matplotlib
pip install torch torchvision torchaudio 
pip install cellpose
```
