from pathlib import Path
import os
import logging

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# IMPORTANT:
# LangChain 1.x uses langchain-classic for MultiQueryRetriever.
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

CHROMA_PATH = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

CHROMA_K = 4


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("MULTI-QUERY RETRIEVAL SYSTEM")
print("=" * 75)


# ============================================================
# CHECK NVIDIA API KEY
# ============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError(
        """
NVIDIA_API_KEY not found.

Please add this to your .env file:

NVIDIA_API_KEY=your_nvidia_api_key
"""
    )

print("✓ NVIDIA API key found")


# ============================================================
# CHECK CHROMA DATABASE
# ============================================================

if not CHROMA_PATH.exists():
    raise FileNotFoundError(
        f"""
ChromaDB was not found.

Expected location:
{CHROMA_PATH}

First create the database by running:

python create_databases.py
"""
    )

print(f"✓ ChromaDB found: {CHROMA_PATH}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("-" * 75)
print("1. LOADING EMBEDDING MODEL")
print("-" * 75)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print(f"✓ Embedding model: {EMBEDDING_MODEL}")


# ============================================================
# LOAD CHROMADB
# ============================================================

print()
print("-" * 75)
print("2. LOADING CHROMADB")
print("-" * 75)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model
)

print(f"✓ Collection: {COLLECTION_NAME}")

# Check number of stored vectors
try:
    document_count = vectorstore._collection.count()
    print(f"✓ Documents/chunks in database: {document_count}")
except Exception:
    print("✓ ChromaDB connected")


# ============================================================
# CREATE BASE RETRIEVER
# ============================================================

print()
print("-" * 75)
print("3. CREATING BASE RETRIEVER")
print("-" * 75)

base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": CHROMA_K
    }
)

print(f"✓ Similarity search enabled")
print(f"✓ Top K: {CHROMA_K}")


# ============================================================
# LOAD NVIDIA NEMOTRON
# ============================================================

print()
print("-" * 75)
print("4. LOADING NVIDIA NEMOTRON")
print("-" * 75)

llm = ChatOpenAI(
    model=NVIDIA_MODEL,
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.2,
    max_tokens=512,
)

print(f"✓ Model: {NVIDIA_MODEL}")


# ============================================================
# CREATE MULTI-QUERY RETRIEVER
# ============================================================

print()
print("-" * 75)
print("5. CREATING MULTI-QUERY RETRIEVER")
print("-" * 75)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True
)

print("✓ MultiQueryRetriever created")
print("✓ Original query will also be searched")


# ============================================================
# GET USER QUESTION
# ============================================================

print()
print("=" * 75)

question = input(
    "Enter your question: "
).strip()

if not question:
    raise ValueError(
        "Question cannot be empty."
    )


# ============================================================
# DISPLAY ORIGINAL QUESTION
# ============================================================

print()
print("=" * 75)
print("ORIGINAL QUESTION")
print("=" * 75)

print(question)


# ============================================================
# GENERATE MULTIPLE QUERIES
# ============================================================

print()
print("=" * 75)
print("GENERATING MULTIPLE QUERY VARIATIONS")
print("=" * 75)

# This lets us inspect the queries generated by the LLM.
try:
    generated_queries = multi_query_retriever.generate_queries(
        question,
        run_manager=None
    )

    for i, generated_query in enumerate(
        generated_queries,
        start=1
    ):
        print()
        print(f"Query {i}:")
        print(generated_query)

except Exception as error:
    print()
    print("Could not display generated queries.")
    print(f"Reason: {error}")

    generated_queries = []


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

print()
print("=" * 75)
print("RETRIEVING DOCUMENTS")
print("=" * 75)

retrieved_docs = multi_query_retriever.invoke(
    question
)


# ============================================================
# DISPLAY RETRIEVAL RESULT
# ============================================================

print()
print("=" * 75)
print("RETRIEVAL COMPLETED")
print("=" * 75)

print(
    f"Unique documents retrieved: "
    f"{len(retrieved_docs)}"
)


# ============================================================
# DISPLAY DOCUMENTS
# ============================================================

for index, document in enumerate(
    retrieved_docs,
    start=1
):

    print()
    print("=" * 75)
    print(f"DOCUMENT {index}")
    print("=" * 75)

    print("\nCONTENT:")
    print("-" * 75)

    print(
        document.page_content[:2000]
    )

    print("\nMETADATA:")
    print("-" * 75)

    print(
        document.metadata
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 75)
print("MULTI-QUERY RETRIEVAL FINISHED")
print("=" * 75)

print(
    """
Pipeline:

User Question
      ↓
Nemotron
      ↓
Multiple Query Variations
      ↓
ChromaDB
      ↓
Similarity Search
      ↓
Duplicate Removal
      ↓
Unique Relevant Documents
"""
)

print("=" * 75)