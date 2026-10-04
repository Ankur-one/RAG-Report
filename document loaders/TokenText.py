from langchain_text_splitters import TextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

spliter = CharacterTextSplitter(
    separator="",
    chunk_size = 1000,
    chunk_overlap=50
)

data = PyPDFLoader("GRU.pdf")
docs = data.load()

chunks = spliter.split_documents(docs)

print("Number of documents:", len(docs))

print(chunks[0].page_content)