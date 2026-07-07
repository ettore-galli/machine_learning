#!/bin/bash

export SERVER_PORT="9876"
export SERVER_HOST="localhost"
export MISTRAL7B_LMS_MODEL_NAME="huggingface/downloaded/mistral-7b-q4_k_m.gguf"

lms load "${MISTRAL7B_LMS_MODEL_NAME}" --gpu max -y --context-length 8192  
lms server start --port "${SERVER_PORT}" --cors --bind "${SERVER_HOST}"