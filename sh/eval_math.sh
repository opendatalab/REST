#!/bin/bash
# set -x

cd $(dirname $(dirname $0))
echo "Current directory: $(pwd)"
MODE=$1
if [ -z "$MODE" ]; then
    MODE="all" # choose from "all", "infer", "eval"
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export COMPASS_DATA_CACHE="data"
MAX_NUM_WORKER=8

source ~/.bashrc
source activate opencompass

for model_abbr in 1.5b 7b 32b; do
    opencompass -r $model_abbr \
        -w outputs/stress_test/math \
        --datasets stress_test_aime2024_gen stress_test_aime2025_gen stress_test_amc23_gen stress_test_gsm8k_gen stress_test_math500_gen \
        --models stress_test_models_$model_abbr \
        --max-num-worker $MAX_NUM_WORKER \
        --summarizer stress_test_math \
        --dump-eval-details False \
        --mode $MODE
done
export DETAILED_THINK_ON="1" # For Nemotron models only
export TEMPERATURE=0.6
export MODEL_NAME="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
opencompass -r 7b \
    -w outputs/stress_test/math \
    --datasets stress_test_aime2024_gen stress_test_aime2025_gen stress_test_amc23_gen stress_test_gsm8k_gen stress_test_math500_gen \
    --models stress_test_models_hf \
    --max-num-worker $MAX_NUM_WORKER \
    --summarizer stress_test_math \
    --dump-eval-details False \
    --mode $MODE
export DETAILED_THINK_ON="0"
export MODEL_NAME="deepseek-r1"
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
opencompass -r api \
    -w outputs/stress_test/math \
    --datasets stress_test_aime2024_gen stress_test_aime2025_gen stress_test_amc23_gen stress_test_gsm8k_gen stress_test_math500_gen \
    --models stress_test_models_api \
    --summarizer stress_test_math \
    --max-num-worker $MAX_NUM_WORKER \
    --dump-eval-details False \
    --mode $MODE
export DETAILED_THINK_ON="0"
export MODEL_NAME="o3-mini"
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
opencompass -r api \
    -w outputs/stress_test/math \
    --datasets stress_test_aime2024_gen stress_test_aime2025_gen stress_test_amc23_gen stress_test_gsm8k_gen stress_test_math500_gen \
    --models stress_test_models_api \
    --summarizer stress_test_math \
    --max-num-worker $MAX_NUM_WORKER \
    --dump-eval-details False \
    --mode $MODE

