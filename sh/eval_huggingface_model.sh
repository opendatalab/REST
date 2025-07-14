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

export DETAILED_THINK_ON="1" # for Nemotron model only
export TP_SIZE=1
export TEMPERATURE=0.6
export MODEL_NAME="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
export COMPASS_DATA_CACHE="data"
huggingface-cli download $MODEL_NAME
opencompass -r hf \
    -w outputs/stress_test/math \
    --datasets stress_test_aime2024_gen stress_test_aime2025_gen stress_test_amc23_gen stress_test_gsm8k_gen stress_test_math500_gen \
    --models stress_test_models_hf \
    --summarizer stress_test_math \
    --max-num-worker $MAX_NUM_WORKER \
    --dump-eval-details False \
    --mode $MODE
export VERIFYER_MODEL_NAME="google/gemma-3-27b-it"
export VERIFYER_API_BASE="http://localhost:62726/v1"
export VERIFYER_API_KEY="EMPTY"
opencompass -r hf \
    -w outputs/stress_test/gpqa \
    --datasets stress_test_gpqa_llmextract_gen \
    --models stress_test_models_hf \
    --summarizer stress_test_gpqa \
    --max-num-worker $MAX_NUM_WORKER \
    --dump-eval-details False \
    --mode $MODE
export COMPASS_DATA_CACHE=""
opencompass -r hf \
    -w outputs/stress_test/code \
    --datasets stress_test_livecodebench_gen \
    --models stress_test_models_hf \
    --max-num-worker $MAX_NUM_WORKER \
    --dump-eval-details False \
    --mode $MODE

export DETAILED_THINK_ON="0"
export TP_SIZE=4
export TEMPERATURE=0.6
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
export COMPASS_DATA_CACHE="data"
huggingface-cli download $MODEL_NAME
opencompass -r hf \
    -w outputs/stress_test/math \
    --datasets stress_test_aime2024_gen stress_test_aime2025_gen stress_test_amc23_gen stress_test_gsm8k_gen stress_test_math500_gen \
    --models stress_test_models_hf \
    --summarizer stress_test_math \
    --max-num-worker $MAX_NUM_WORKER \
    --dump-eval-details False \
    --mode $MODE
export VERIFYER_MODEL_NAME="google/gemma-3-27b-it"
export VERIFYER_API_BASE="http://localhost:62726/v1"
export VERIFYER_API_KEY="EMPTY"
opencompass -r hf \
    -w outputs/stress_test/gpqa \
    --datasets stress_test_gpqa_llmextract_gen \
    --models stress_test_models_hf \
    --summarizer stress_test_gpqa \
    --max-num-worker $MAX_NUM_WORKER \
    --dump-eval-details False \
    --mode $MODE
export COMPASS_DATA_CACHE=""
opencompass -r hf \
    -w outputs/stress_test/code \
    --datasets stress_test_livecodebench_gen \
    --models stress_test_models_hf \
    --max-num-worker $MAX_NUM_WORKER \
    --dump-eval-details False \
    --mode $MODE

