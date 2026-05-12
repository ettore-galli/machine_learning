# https://ollama.com/download
#
source ./ollama-env.sh

#!/bin/bash

json=$(jq -n \
  --arg prompt "$1" \
  '{model:"qwen-custom-1", prompt:$prompt, stream:false}')

curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d "$json"
 
