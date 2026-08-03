# https://ollama.com/download
#
source ./env.sh

echo OLLAMA_MODELS=${OLLAMA_MODELS}

${EXECUTABLE} pull qwen2.5-coder:7b


