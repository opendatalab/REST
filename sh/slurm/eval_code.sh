#!/bin/bash
# set -x

cd $(dirname $(dirname $0))
echo "Current directory: $(pwd)"
MODE=$1
if [ -z "$MODE" ]; then
    MODE="all"
fi
PARTITION=$2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
unset COMPASS_DATA_CACHE
MAX_NUM_WORKER=8

source ~/.bashrc
source activate opencompass

for model_abbr in 1.5b 7b 32b; do
    opencompass  --slurm \
        -p $PARTITION \
        -q auto \
        -r $model_abbr \
        -w outputs/stress_test/code \
        --datasets stress_test_livecodebench_gen \
        --models stress_test_models_$model_abbr \
        --max-num-worker $MAX_NUM_WORKER \
        --dump-eval-details False \
        --mode $MODE
done
