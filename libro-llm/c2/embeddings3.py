import os
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Generate embeddings for several sentences
sentences = [
    "The dog plays in the garden.",
    "The puppy runs through the park.",
    "Artificial intelligence is transforming industry.",
    "Language models process text sequences."
]

embeddings = [generate_embedding(s) for s in sentences]

print(f"Dimensions of each embedding: {len(embeddings[0])}")
# text-embedding-3-small produces vectors of 1536 dimensions