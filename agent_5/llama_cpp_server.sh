if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

python3 -m llama_cpp.server \
  --model "${MODEL_MISTRAL7B_MODEL}"  \
  --host "${LLAMA_CPP_SERVER_HOST}" \
  --port "${LLAMA_CPP_SERVER_PORT}"