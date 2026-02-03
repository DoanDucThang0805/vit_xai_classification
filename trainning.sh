#!/bin/bash

# load conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /media/icnlab/Data/Thang/plan_dieases/env
# Di chuyển vào thư mục src
cd src

# Chạy script train
# PYTHONPATH=src python -m trainning.resnet50_train
# PYTHONPATH=src python -m trainning.resnet50_train
PYTHONPATH=src python -m trainning.resnet50_train

PYTHONPATH=src python -m trainning.vgg16_train
PYTHONPATH=src python -m trainning.vgg16_train
# PYTHONPATH=src python -m trainning.vgg16_train

PYTHONPATH=src python -m trainning.shuffelnetv2_train
PYTHONPATH=src python -m trainning.shuffelnetv2_train
PYTHONPATH=src python -m trainning.shuffelnetv2_train

PYTHONPATH=src python -m trainning.squezzenet_train
PYTHONPATH=src python -m trainning.squezzenet_train
PYTHONPATH=src python -m trainning.squezzenet_train
