#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# List of parameters for serialization
params=("fixed" "random" "valid" "plain" "json" "pairwise" "span")

# Loop over the parameters
for param in "${params[@]}"; do
    echo "Running script with stype: $param"
    python train_ditto.py --task d1 --batch_size 128 --max_len 256 --lr 3e-5 --n_epochs 20 --finetuning --lm roberta --stype $param
done

# List of parameters for datasets
params=("d1" "d2" "d3" "d4" "d5" "d6")

# Loop over the parameters
for param in "${params[@]}"; do
    echo "Running script with stype: $param"
    python train_ditto.py --task $param --batch_size 128 --max_len 256 --lr 3e-5 --n_epochs 20 --finetuning --lm roberta --stype span --dk general
done