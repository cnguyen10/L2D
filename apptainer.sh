#!/bin/sh
DEVICE_ID=1  # which GPU is going to be used
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

# to build an image, use the following command
# >> apptainer build apptainer.sif apptainer.def

apptainer exec --nv --bind /sda2/datasets:/sda2/datasets apptainer.sif /app/venv/bin/python3 "main.py"
apptainer exec --nv --bind /sda2/datasets:/sda2/datasets apptainer.sif /app/venv/bin/python3 "main.py" "hparams.epsilon_upper=[0.15, 0.15, 0.15, 0.15, 0.4]"
apptainer exec --nv --bind /sda2/datasets:/sda2/datasets apptainer.sif /app/venv/bin/python3 "main.py" "hparams.epsilon_upper=[0.1, 0.1, 0.1, 0.1, 0.6]"
apptainer exec --nv --bind /sda2/datasets:/sda2/datasets apptainer.sif /app/venv/bin/python3 "main.py" "hparams.epsilon_upper=[0.05, 0.05, 0.05, 0.05, 0.8]"