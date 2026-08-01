# https://ollama.com/download
#
source ./env.sh

#!/bin/bash

# endpoint="http://localhost:11434/api/generate";
endpoint="http://localhost:11434/v1/completions";

json=$(jq -n \
  --arg prompt "$1" \
  '{
  model: "Qwen2.5-Coder", 
  prompt: $prompt, 
  stream: false,
  max_tokens: 512
  }')

echo $json

curl "${endpoint}" \
  -H "Content-Type: application/json" \
  -d "$json"
 
