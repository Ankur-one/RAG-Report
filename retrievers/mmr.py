from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = [
    Document(
        page_content=(
            "Gradient descent is an optimization algorithm used in machine learning."
        )
    ),
    Document(
        page_content=(
            "Gradient descent minimizes the loss function."
        )
    ),
    Document(
        page_content=("Gradient descent updates model parameters iteratively.")
    ),
    Document(
        page_content=(
            "Gradient descent can be used for linear regression and neural networks."
        )
    ),
    Document(
        page_content=(
            "Neural networks are a type of machine learning model inspired by the human brain."
        )
    ),
    Document(
        page_content=(
            "Deep Learning is a subset of Machine Learning."
        )
    )
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("\n=======Similarity Search Results=======\n")

similarity_docs = similarity_retriever.invoke("What is Gradient Descent?")

for doc in similarity_docs:
    print(doc.page_content)
    print("-" * 70)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

print("\n=======MMR Search Results=======\n")

mmr_docs = mmr_retriever.invoke("What is Gradient Descent?")

for doc in mmr_docs:
    print(doc.page_content)
    print("-" * 70)