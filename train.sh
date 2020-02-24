#!/bin/bash

cats=(bottle capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transitor wood zipper)
models=(Bottleneckv5)
for cat in "${cats[@]}"
do
    
    for model in "${models[@]}"
    do
        echo ==========MODEL: ${model} Cateogry: ${cat}====================
        CUDA_VISIBILE_DEVICES=0,1 python main.py --batch_size 32 --model ${model} --exp_name whole_${model}_${cat} --seed 42 --crop_size 256 --category ${cat} --train_mode whole_image --epochs 1
        CUDA_VISIBILE_DEVICES=0,1 python main.py --batch_size 32 --model ${model} --exp_name patch_${model}_${cat} --seed 42 --crop_size 128 --category ${cat} --train_mode patch --epochs 1
    done
done


cats=(bottle capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transitor wood zipper)
models=(Bottleneckv4 Bottleneckv2)
for cat in "${cats[@]}"
do
    
    for model in "${models[@]}"
    do
        echo ==========MODEL: ${model} Cateogry: ${cat}====================
        CUDA_VISIBILE_DEVICES=0,1 python main.py --batch_size 32 --model ${model} --exp_name brute_${model}_${cat} --seed 42 --crop_size 256 --category ${cat} --epochs 100
    done
done


# for q in (bottle capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transitor wood zipper)
# do
#     CUDA_VISIBLE_DEVICES=0 python eval-object-detection.py --dataset CityScapes_seg --q $q
# done
# CUDA_VISIBILE_DEVICES=0,1 python main.py --batch_size 32 --model Bottleneckv4 --exp_name BottleNeckv4_wood --seed 42 --crop_size 300 --category wood

# CUDA_VISIBILE_DEVICES=0,1 python main.py  --model Bottleneckv5  --batch_size 32 --exp_name BottleNeckv_5 --seed 42 --crop_size 128 --category bottle --train_mode patch