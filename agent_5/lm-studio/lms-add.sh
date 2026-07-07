#!/bin/bash
export MISTRAL7B_LMS_MODEL_NAME="huggingface/downloaded/mistral-7b-q4_k_m.gguf"
lms load "${MISTRAL7B_LMS_MODEL_NAME}" --gpu max -y --context-length 8192  