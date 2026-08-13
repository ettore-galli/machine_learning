# LM STUDIO

## Config

```sh
# Maybe create missing part 
cd /Users/ettoregalli/Library/Application Support/lm-studio

vi settings.json
```

settings.json

```json
{
  "modelLibraryPath": "/Volumes/DOCKER/huggingface/gguf-models"
}
```

Reporganize

```sh
mkdir -p /Volumes/DOCKER/huggingface/gguf-models/Meta-Llama-3.1-8B-Instruct-Q5_K_M
mkdir -p /Volumes/DOCKER/huggingface/gguf-models/mistral-7b-Q4_K_M
mkdir -p /Volumes/DOCKER/huggingface/gguf-models/qwen2.5-1.5b-instruct-q4_k_m
mkdir -p /Volumes/DOCKER/huggingface/gguf-models/qwen2.5-3b-instruct-q8_0
mkdir -p /Volumes/DOCKER/huggingface/gguf-models/Qwen2.5-Coder-7B.Q4_K_M
mkdir -p /Volumes/DOCKER/huggingface/gguf-models/Qwen2.5-Coder-14B-Q5_K_M

mv /Volumes/DOCKER/huggingface/gguf-models/Meta-Llama-3.1-8B-Instruct-Q5  /Volumes/DOCKER/huggingface/gguf-models/Meta-Llama-3.1-8B-Instruct-Q5_K_M
mv /Volumes/DOCKER/huggingface/gguf-models/mistral-7b-Q4_K_M.gguf /Volumes/DOCKER/huggingface/gguf-models/mistral-7b-Q4_K_M
mv /Volumes/DOCKER/huggingface/gguf-models/qwen2.5-1.5b-instruct-q4_k_m. /Volumes/DOCKER/huggingface/gguf-models/qwen2.5-1.5b-instruct-q4_k_m
mv /Volumes/DOCKER/huggingface/gguf-models/qwen2.5-3b-instruct-q8_0.gguf /Volumes/DOCKER/huggingface/gguf-models/qwen2.5-3b-instruct-q8_0
mv /Volumes/DOCKER/huggingface/gguf-models/Qwen2.5-Coder-7B.Q4_K_M.gguf /Volumes/DOCKER/huggingface/gguf-models/Qwen2.5-Coder-7B.Q4_K_M
mv /Volumes/DOCKER/huggingface/gguf-models/Qwen2.5-Coder-14B-Q5_K_M.gguf /Volumes/DOCKER/huggingface/gguf-models/Qwen2.5-Coder-14B-Q5_K_M
```
