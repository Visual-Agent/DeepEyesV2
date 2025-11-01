#!/bin/bash
mkdir /root/LMUData
mkdir /root/LMUData/images

MODEL="DeepEyesV2-vllm" 
DATASETS=("VStarBench" "HRBench4K" "HRBench8K" "MME-RealWorld-Lite" "TreeBench" "OCRBench" "SEEDBench2_Plus" "CharXiv_descriptive_val" "CharXiv_reasoning_val" "ChartQA_TEST" "MME-RealWorld")
MATH_DATASETS=("MathVista_MINI" "MathVerse_MINI" "MathVision" "WeMath" "DynaMath" "LogicVista")
INFERENCE_MODE="agent"
API_NPROC=16
MODEL_CONFIGS='{"DeepEyesV2-vllm": {"api_base": "http://10.39.6.41:28000,http://10.39.6.41:18000", "max_tokens": 20480}}'
SAVE_NAME="deepeyesv2"

python run.py \
    --model "$MODEL" \
    --work-dir "./results/$SAVE_NAME" \
    --data "${DATASETS[@]}" \
    --inference-mode "$INFERENCE_MODE" \
    --api-nproc "$API_NPROC" \
    --model-configs "$MODEL_CONFIGS" \
    --reuse \
    --judge llm-judge 

python run.py \
    --model "$MODEL" \
    --work-dir "./results/$SAVE_NAME" \
    --data "${MATH_DATASETS[@]}" \
    --inference-mode "$INFERENCE_MODE" \
    --api-nproc "$API_NPROC" \
    --model-configs "$MODEL_CONFIGS" \
    --reuse \
    --judge llm-judge


