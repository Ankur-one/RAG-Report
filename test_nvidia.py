
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env
load_dotenv()

# Get NVIDIA API key
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError(
        "NVIDIA_API_KEY not found in .env file."
    )

# NVIDIA Nemotron
model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.2,
    max_tokens=512,
)


# Test request
response = model.invoke(
    "Explain Retrieval Augmented Generation (RAG) in simple terms."
)

print("\n" + "=" * 60)
print("NVIDIA NEMOTRON RESPONSE")
print("=" * 60)
print(response.content)
print("=" * 60)
