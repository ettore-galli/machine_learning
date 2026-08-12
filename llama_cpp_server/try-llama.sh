curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-needed" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Ciao"}]
  }'