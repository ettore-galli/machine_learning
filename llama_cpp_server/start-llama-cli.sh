export MODEL="/Volumes/DOCKER/huggingface/gguf-models/Qwen2.5-Coder-7B.Q4_K_M.gguf"
export LLAMA_CLI_EXE="./llama-b10373/llama-cli"

"${LLAMA_CLI_EXE}" \
  --model "${MODEL}"  