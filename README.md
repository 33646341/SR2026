# SR2026

This repository contains the official PyTorch implementation of our paper: **Isotropic Reconstruction of Anisotropic vEM Volumes with ViT-Guided Diffusion**.

## Project Overview

SR2026 is a volumetric electron microscopy (vEM) anisotropic data reconstruction tool based on diffusion models. This project can reconstruct low-resolution, anisotropic vEM volume data into high-quality isotropic 3D volumes, with important applications in neuroscience, biology, and other fields.

## Core Features

- **Anisotropic to Isotropic Reconstruction**: Supports vEM data reconstruction with various anisotropy ratios
- **Multiple Model Configurations**: Provides various model configurations including vEMDiffuse-i, vEMDiffuse-a, EMDiffuse-n, EMDiffuse-r
- **DDIM Sampling**: Supports fast DDIM sampling, balancing reconstruction quality and inference speed
- **Multi-GPU Distributed Training**: Supports parallel training using multiple GPUs
- **Visualization and Monitoring**: Integrated TensorBoard for training process visualization

## Technology Stack

- **Deep Learning Framework**: PyTorch >= 1.13.0
- **Python Version**: 3.12+
- **Package Manager**: uv
- **Core Model**: U-Net + Guided Diffusion
- **Other Dependencies**:
  - Data Processing: tifffile, imagecodecs, numpy, scipy
  - Visualization: tensorboard, matplotlib
  - Monitoring: nvitop
  - Image Processing: opencv-python, imutils

## Architecture Highlights

- **Modular Design**: Models, datasets, loss functions, etc., adopt modular design for easy extension
- **Configuration-Driven**: Uses JSON configuration files to manage training and testing parameters
- **EMA Smoothing**: Uses Exponential Moving Average (EMA) to optimize model weights
- **Flexible Data Loading**: Supports custom datasets and data augmentation

## Common Commands

```bash
# Train model - Multi-GPU training with vEMDiffuse-i configuration
python run.py -c config/vEMDiffuse-i.json -b 16 --gpu 0,1,2,3 --port 20022 --path ./vEM_data -z 6 --lr 5e-5

# Test model - Load pretrained weights for anisotropic reconstruction
python run.py -p test -c config/vEMDiffuse-i.json --gpu 0 -b 16 --path ./vEM_test_data -z 6 --resume ./experiments/vEMDiffuse-i/best --mean 1 --step 200

# Visualize training - Monitor training process with TensorBoard
uv run tensorboard --logdir .
```

## Project Structure

- `config/` - Model configuration files directory
- `models/` - Network model definitions (including U-Net, diffusion models, etc.)
- `data/` - Datasets and data loaders
- `core/` - Core utilities (logging, configuration parsing, etc.)
- `3D-SR-Unet/` - 3D super-resolution U-Net module

## Dependency

Please install PyTorch before you run the code. We strongly recommend you install uv, where we use Python 3.12.

```bash
uv sync
```

## Data Preparation

Download or prepare your vEM training data. The training file structure should be like:

```
vEM_data
	0.tif // The first layer
	1.tif // The second layer
	...
	n.tif // The (n+1)th layer
```

## Training

```bash
python run.py -c config/dinov3loss.json -b 16 --gpu 0,1,2,3 --port 20022 --path ./vEM_data -z 6 --lr 5e-5
```

`-z` means the subsampling factor of the Z axis. For example, to reconstruct an 8 nm x 8 nm x 8 nm volume from an 8 nm x 8 nm x 48 nm volume, the subsampling factor should be 6.

## Testing

To test, prepare an anisotropic volume.

```
vEM_test_data
	0.tif // The first layer
	1.tif // The second layer
	...
	n.tif // The (n+1)th layer
```

```bash
python run.py -p test -c config/dinov3loss.json --gpu 0 -b 16 --path ./vEM_test_data/ -z 6 --resume ./experiments/dinov3loss/best --mean 1 --step 200
```

Adjust the model weight directory to where your best model weights are saved.

All results, including training and inference, will be stored in a newly created folder under `./experiments`.

Running the diffusion process on a **GPU** is highly recommended for both training and testing.

Should you have any questions regarding the code, please do not hesitate to contact us.
