#!/bin/sh
DEVICE_ID=1  # which GPU is going to be used
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

# to build an image, use the following command
# >> apptainer build apptainer.sif apptainer.def

python3 "main.py"