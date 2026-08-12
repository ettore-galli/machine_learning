export MODEL="/Volumes/DOCKER/huggingface/gguf-models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
export HOST=0.0.0.0
export PORT=8000
export LLAMA_SERVER_EXE="./llama-b10373/llama-server"

"${LLAMA_SERVER_EXE}" \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key not-needed 