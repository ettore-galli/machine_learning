from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import GenerationConfig, pipeline

# 1. Load .env file (contains your HF_TOKEN)
load_dotenv()

# 1. / bis show the current env
for key, value in os.environ.items():
    if key.startswith("HF_"):
        print(key, value)

if not os.getenv("HF_HOME"):
    raise SystemExit("❌ Define root directory for model (HF_HOME)")

# 2. Log in to Hugging Face with your token
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise SystemExit(
        "❌ No Hugging Face token found. Add HF_TOKEN=... in your .env file."
    )
login(token=hf_token)

# 3. Choose a small, free text-generation model
MODEL_ID = "google/flan-t5-base"


# 4. Create a local text2text-generation pipeline
pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
)

print("✅ Agent ready! Type a question, or 'quit' to stop.\n")

while True:
    try:
        user_input = input("> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        result = pipe(user_input, max_new_tokens=100)[0]["generated_text"]
        print(f"\n🤖 Agent: {result}\n")

    except KeyboardInterrupt:
        print("\n(Stopped) Goodbye!")
        break
