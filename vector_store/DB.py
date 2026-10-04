from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# DB.py location:
# RAG Report/vector_store/DB.py

CURRENT_DIR = Path(__file__).resolve().parent

# RAG Report/
PROJECT_ROOT = CURRENT_DIR.parent

# RAG Report/chroma_db/
CHROMA_PATH = PROJECT_ROOT / "chroma_db"


print("=" * 60)
print("CHROMA DATABASE PATH")
print("=" * 60)
print(CHROMA_PATH)


# ============================================================
# 3. CREATE DOCUMENTS
# ============================================================

docs = [
    Document(
        page_content=(
            "Artificial Intelligence is a branch of computer science "
            "that aims to create machines capable of intelligent behavior."
        ),
        metadata={
            "source": "first_document"
        }
    ),

    Document(
        page_content=(
            "Machine Learning is a subset of Artificial Intelligence."
        ),
        metadata={
            "source": "second_document"
        }
    ),

    Document(
        page_content=(
            "Deep Learning is a subset of Machine Learning."
        ),
        metadata={
            "source": "third_document"
        }
    )
]


# ============================================================
# 4. CREATE EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# 5. CREATE CHROMA VECTOR STORE
# ============================================================

print("\n" + "=" * 60)
print("CREATING CHROMA VECTOR STORE")
print("=" * 60)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=str(CHROMA_PATH),
    collection_name="rag_documents"
)

print("Documents stored successfully.")
print(f"Database location: {CHROMA_PATH}")


# ============================================================
# 6. SIMILARITY SEARCH
# ============================================================

print("\n" + "=" * 60)
print("SIMILARITY SEARCH")
print("=" * 60)

query = "What is used for data analysis?"

results = vector_store.similarity_search(
    query,
    k=2
)

for result in results:

    print("\nPage Content:")
    print(result.page_content)

    print("\nMetadata:")
    print(result.metadata)

    print("-" * 50)


# ============================================================
# 7. CREATE RETRIEVER
# ============================================================

print("\n" + "=" * 60)
print("CREATING RETRIEVER")
print("=" * 60)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2
    }
)


# ============================================================
# 8. RETRIEVE DOCUMENTS
# ============================================================

query = "What is used for data analysis?"

retrieved_docs = retriever.invoke(query)


print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)

for doc in retrieved_docs:

    print("\nPage Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    print("-" * 50)


# ============================================================
# 9. DATABASE COUNT
# ============================================================

print("\n" + "=" * 60)
print("DATABASE INFORMATION")
print("=" * 60)

print(
    "Number of stored documents:",
    vector_store._collection.count()
)

print("\nChromaDB setup completed successfully!")