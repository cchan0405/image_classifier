# Image Classifier

This project is a simple image classification model built using **PyTorch** and **timm**. It trains on a dataset of images and predicts their classes using a deep learning architecture.

## Features
- Supports custom datasets using `ImageFolder`
- Training loop tracks **average loss per epoch**
- Optional “pseudo-validation” to visualize loss trends
- Uses **Matplotlib** for plotting training progress

## Requirements
- Python 3.11+
- PyTorch
- torchvision
- timm
- matplotlib

## Usage
1. Prepare your dataset in an `ImageFolder`-compatible format.
2. Define transforms and DataLoader.
3. Train the model and track loss per epoch.
4. Optionally, evaluate on a validation set or compute pseudo-validation loss.
5. Plot training and validation loss to monitor performance.

## License
MIT License
