#!#!/bin/bash

# load conda vào shell
conda init
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /media/icnlab/Data/Thang/plan_dieases/env

# Di chuyển vào thư mục src
cd src

# Chạy script train
PYTHONPATH=src python -m trainning.train
