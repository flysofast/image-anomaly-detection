#!/bin/bash
# cats=(bottle capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transitor wood zipper)
# models=(Bottleneckv4 Bottleneckv2)
# for cat in "${cats[@]}"
# do
    
#     for model in "${models[@]}"
#     do
#         echo ==========MODEL: ${model} Cateogry: ${cat}====================
#         CUDA_VISIBILE_DEVICES=0,1 python main.py --batch_size 32 --model ${model} --exp_name brute_${model}_${cat} --seed 42 --crop_size 300 --category ${cat}
#     done
# done


# for q in (bottle capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transitor wood zipper)
# do
#     CUDA_VISIBLE_DEVICES=0 python eval-object-detection.py --dataset CityScapes_seg --q $q
# done
# CUDA_VISIBILE_DEVICES=0,1 python main.py --batch_size 32 --model Bottleneckv4 --exp_name BottleNeckv4_wood --seed 42 --crop_size 300 --category wood

CUDA_VISIBILE_DEVICES=0,1 python main.py --batch_size 32 --exp_name BottleNeck_5 --seed 42 --crop_size 128 --category zipper