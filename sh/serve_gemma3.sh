#!/bin/bash
# set -x

source ~/.bashrc
source activate sglang044
lsof -i :62726 | awk 'NR!=1 {print $2}' | xargs -r kill -9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
N_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "N_GPU: $N_GPU"

MODEL_NAME_OR_PATH="google/gemma-3-27b-it"
python -m sglang.launch_server \
    --model-path $MODEL_NAME_OR_PATH \
    --dp 1 \
    --tp $N_GPU \
    --disable-cuda-graph \
    --host 0.0.0.0 \
    --port 62726