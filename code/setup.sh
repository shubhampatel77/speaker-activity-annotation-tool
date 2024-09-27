#!/bin/bash

# Install RetinaFace package
pip install retina-face

# Install tf-keras package
pip install tf-keras

# Uninstall any existing version of TensorFlow
pip uninstall -y tensorflow

# Install TensorFlow version 2.15.0
pip install tensorflow==2.15.0

#goturn-pytorch specific imports
pip install loguru
pip install torchsummary
