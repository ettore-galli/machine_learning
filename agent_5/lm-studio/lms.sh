#!/bin/bash

SERVER_PORT="9876"
SERVER_HOST="localhost"
MISTRAL7B_LMS_MODEL_NAME=""huggingface/downloaded/mistral-7b-q4_k_m.gguf""

lms load "${MODEL_MISTRAL7B_LMS_MODEL_NAME}" --gpu max -y   # sostituisci con il tuo modello
lms server start --port "${SERVER_PORT}" --cors --bind "${SERVER_HOST}"