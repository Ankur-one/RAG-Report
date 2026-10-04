from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro-14-and-16/"

loader = WebBaseLoader([url])

# Load document
docs = loader.load()
