# ============================================================
# RAG VECTOR DATABASE CREATION
#
# 1. Load PDF
# 2. Split into chunks
# 3. Create free local embeddings
# 4. Store embeddings in ChromaDB
# ============================================================

from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. PROJECT PATH
# ============================================================

# create_databases.py is directly inside:
#
# RAG Report/
# └── create_databases.py

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 3. PDF PATH
# ============================================================

# Change "DL.pdf" if your actual PDF has another name.

PDF_PATH = BASE_DIR / "document loaders" / "DL.pdf"


# ============================================================
# 4. CHROMA DATABASE PATH
# ============================================================

# ChromaDB will be created here:
#
# RAG Report/
# └── chroma_db/

CHROMA_PATH = BASE_DIR / "chroma_db"


# ============================================================
# 5. CHECK PDF
# ============================================================

print("=" * 70)
print("RAG VECTOR DATABASE CREATION")
print("=" * 70)

print(f"Project directory : {BASE_DIR}")
print(f"PDF path          : {PDF_PATH}")
print(f"ChromaDB path     : {CHROMA_PATH}")


if not PDF_PATH.exists():

    raise FileNotFoundError(
        f"\nPDF not found:\n{PDF_PATH}\n\n"
        "Please check:\n"
        "1. The PDF is inside the 'document loaders' folder.\n"
        "2. The PDF filename is exactly 'DL.pdf'."
    )


# ============================================================
# 6. LOAD PDF
# ============================================================

print("\n" + "=" * 70)
print("1. LOADING PDF")
print("=" * 70)

loader = PyPDFLoader(
    str(PDF_PATH)
)

docs = loader.load()

print(f"PDF loaded successfully.")
print(f"Number of pages: {len(docs)}")


# ============================================================
# 7. SPLIT DOCUMENT
# ============================================================

print("\n" + "=" * 70)
print("2. SPLITTING DOCUMENT")
print("=" * 70)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(
    docs
)

print(f"Number of chunks: {len(chunks)}")


# ============================================================
# 8. CREATE FREE EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 70)
print("3. CREATING EMBEDDINGS")
print("=" * 70)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# 9. CREATE CHROMADB
# ============================================================

print("\n" + "=" * 70)
print("4. CREATING CHROMADB")
print("=" * 70)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(CHROMA_PATH),
    collection_name="rag_documents"
)

print("Documents stored successfully in ChromaDB.")


# ============================================================
# 10. VERIFY DATABASE
# ============================================================

print("\n" + "=" * 70)
print("DATABASE INFORMATION")
print("=" * 70)

print(f"Database location : {CHROMA_PATH}")

print(
    f"Stored chunks     : {vectorstore._collection.count()}"
)


# ============================================================
# 11. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("Vector database created successfully!")
print("PDF → Chunks → Embeddings → ChromaDB")

print("=" * 70)