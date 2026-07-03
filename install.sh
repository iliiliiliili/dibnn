#!/bin/bash -i

conda create -n dbnn python=3.13

conda activate dbnn

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt