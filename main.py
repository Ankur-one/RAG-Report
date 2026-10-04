import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "rag_documents"


# ============================================================
# 3. CHECK NVIDIA API KEY
# ============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError(
        """
NVIDIA_API_KEY was not found.

Please add the following to your .env file:

NVIDIA_API_KEY=your_nvidia_api_key
"""
    )


# ============================================================
# 4. CHECK CHROMADB
# ============================================================

if not CHROMA_PATH.exists():
    raise FileNotFoundError(
        f"""
ChromaDB was not found.

Expected location:

{CHROMA_PATH}

Please run:

python create_databases.py
"""
    )


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# 6. LOAD CHROMADB
# ============================================================

print()
print("=" * 70)
print("LOADING CHROMADB")
print("=" * 70)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model
)

print("ChromaDB loaded successfully.")

try:
    collection_count = vectorstore._collection.count()
    print(f"Stored chunks: {collection_count}")
except Exception:
    print("Could not determine collection size.")


# ============================================================
# 7. CREATE BASE RETRIEVER
# ============================================================

print()
print("=" * 70)
print("CREATING BASE RETRIEVER")
print("=" * 70)

base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)

print("Base similarity retriever created.")
print("Top K = 4")


# ============================================================
# 8. INITIALIZE NVIDIA NEMOTRON
# ============================================================

print()
print("=" * 70)
print("LOADING NVIDIA NEMOTRON")
print("=" * 70)

llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.2,
    max_tokens=1000,
)

print("NVIDIA Nemotron loaded successfully.")


# ============================================================
# 9. CREATE MULTI-QUERY RETRIEVER
# ============================================================

print()
print("=" * 70)
print("CREATING MULTI-QUERY RETRIEVER")
print("=" * 70)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True
)

print("Multi-Query Retriever created successfully.")


# ============================================================
# 10. RAG PROMPT
# ============================================================

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant for a document-based
Retrieval-Augmented Generation (RAG) system.

Your task is to answer the user's question using ONLY
the information provided in the context.

Rules:

1. Use only the provided context.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer cannot be found in the context,
   say exactly:

"I could not find the answer in the provided context."

5. Give a clear and understandable answer.
6. When possible, explain concepts with examples.
7. Keep the answer focused on the user's question.
"""
        ),
        (
            "human",
            """
Context:

{context}


Question:

{question}


Answer:
"""
        ),
    ]
)


# ============================================================
# 11. RAG SYSTEM READY
# ============================================================

print()
print("=" * 70)
print("RAG SYSTEM INITIALIZED SUCCESSFULLY")
print("=" * 70)

print(
    """
Pipeline:

User Question
      ↓
Multi-Query Generation
      ↓
Multiple Search Queries
      ↓
ChromaDB
      ↓
Relevant Documents
      ↓
Context
      ↓
RAG Prompt
      ↓
NVIDIA Nemotron
      ↓
Final Answer
"""
)

print("Press 0 to exit.")


# ============================================================
# 12. QUESTION-ANSWER LOOP
# ============================================================

while True:

    print()
    user_query = input("Enter your question: ").strip()

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if user_query == "0":
        print("\nExiting the program.")
        break

    # --------------------------------------------------------
    # Empty question
    # --------------------------------------------------------

    if not user_query:
        print("Please enter a question.")
        continue

    # --------------------------------------------------------
    # Multi-query retrieval
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("RETRIEVING RELEVANT DOCUMENTS...")
    print("-" * 70)

    try:

        retrieved_docs = multi_query_retriever.invoke(
            user_query
        )

    except Exception as error:

        print()
        print("Retrieval error:")
        print(error)
        continue

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not retrieved_docs:

        print(
            "\nNo relevant documents were found."
        )

        continue

    print(
        f"\nRetrieved documents: "
        f"{len(retrieved_docs)}"
    )

    # --------------------------------------------------------
    # Combine retrieved documents
    # --------------------------------------------------------

    combined_context = "\n\n".join(
        [
            doc.page_content
            for doc in retrieved_docs
        ]
    )

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    formatted_prompt = prompt_template.format_prompt(
        context=combined_context,
        question=user_query
    )

    # --------------------------------------------------------
    # Generate final answer
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("GENERATING ANSWER...")
    print("-" * 70)

    try:

        response = llm.invoke(
            formatted_prompt
        )

        answer = response.content

    except Exception as error:

        print()
        print("LLM error:")
        print(error)
        continue

    # --------------------------------------------------------
    # Display answer
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    # --------------------------------------------------------
    # Display source information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for index, document in enumerate(
        retrieved_docs,
        start=1
    ):

        metadata = document.metadata

        print(
            f"\nSource {index}:"
        )

        if metadata:

            for key, value in metadata.items():

                print(
                    f"  {key}: {value}"
                )

        else:

            print(
                "  No metadata available."
            )

    print()
    print("=" * 70)