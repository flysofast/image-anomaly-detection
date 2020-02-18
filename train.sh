#!/bin/bash
CUDA_VISIBILE_DEVICES=0,1 python main.py --batch_size 32 --exp_name BottleNeck_4 --seed 42 --crop_size 300 --category hazelnut
