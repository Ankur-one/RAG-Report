from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("GRU.pdf")

# Load document
docs = data.load()
print(docs)
print(len(docs))