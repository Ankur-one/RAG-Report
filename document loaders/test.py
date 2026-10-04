from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

spliter = CharacterTextSplitter(
    separator="",
    chunk_size = 10,
    chunk_overlap=1
)



# Create loader
loader = TextLoader("notes.txt")

# Load document
documents = loader.load()

chunks = spliter.split_documents(documents)


for i in chunks:
    print(i.page_content)
    print()
    print()
    print()



print(chunks)
print(len(chunks))


print("Number of documents:", len(documents))

print("\nDocument Content:")
print("-" * 50)

print(documents[0].page_content)

print("\nMetadata:")
print("-" * 50)

print(documents[0].metadata)