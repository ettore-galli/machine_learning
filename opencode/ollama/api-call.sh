# https://ollama.com/download
#
source ./env.sh

#!/bin/bash

json=$(jq -n \
  --arg prompt "$1" \
  '{model:"Qwen2.5-Coder", prompt:$prompt, stream:false}')

curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d "$json"
 
