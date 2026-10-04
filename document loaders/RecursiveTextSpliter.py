from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load PDF
loader = PyPDFLoader("GRU.pdf")

docs = loader.load()

print("Number of pages:", len(docs))


# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Number of chunks:", len(chunks))


# Check chunk sizes
for i, chunk in enumerate(chunks[:5]):

    print(
        f"Chunk {i + 1}: "
        f"{len(chunk.page_content)} characters"
    )

    print(chunk.page_content[:100])
    print("-" * 50)