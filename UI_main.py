import os
import hashlib
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Book RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_BASE_PATH = BASE_DIR / "chroma_db"

UPLOAD_DIR = BASE_DIR / "uploaded_documents"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHROMA_BASE_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. CONFIGURATION
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

NVIDIA_MODEL = (
    "nvidia/nemotron-3.5-lightning-30b-a3b"
)

NVIDIA_BASE_URL = (
    "https://integrate.api.nvidia.com/v1"
)

COLLECTION_NAME = "rag_documents"

# Number of documents/chunks retrieved
TOP_K = 5


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .source-card {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 AI Book RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Upload a PDF book and ask questions. The AI answers
    using only the uploaded document and shows the
    relevant PDF page numbers.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "document_count" not in st.session_state:
    st.session_state.document_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 8. API KEY
# ============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    st.markdown("---")

    st.subheader("📄 Upload Book")

    uploaded_file = st.file_uploader(
        "Upload your PDF book",
        type=["pdf"],
        help="Upload a PDF book/document.",
    )

    st.markdown("---")

    st.subheader("🧠 AI Configuration")

    st.write("**LLM:** NVIDIA Nemotron")

    st.write(
        "**Embeddings:** "
        "all-MiniLM-L6-v2"
    )

    st.write("**Vector DB:** ChromaDB")

    st.write("**Retriever:** Multi-Query")

    st.write(
        f"**Top K:** {TOP_K}"
    )

    st.markdown("---")

    st.subheader("📊 Current Book")

    if st.session_state.document_name:

        st.success(
            st.session_state.document_name
        )

        st.write(
            f"**Chunks:** "
            f"{st.session_state.document_count}"
        )

    else:

        st.info(
            "No PDF uploaded yet."
        )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Current Book",
        use_container_width=True,
    ):

        st.session_state.vectorstore = None

        st.session_state.retriever = None

        st.session_state.document_name = None

        st.session_state.document_hash = None

        st.session_state.document_count = 0

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 10. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# 11. LOAD NVIDIA NEMOTRON
# ============================================================

@st.cache_resource
def load_llm(api_key):

    return ChatOpenAI(
        model=NVIDIA_MODEL,
        api_key=api_key,
        base_url=NVIDIA_BASE_URL,
        temperature=0.1,
        max_tokens=1500,
    )


# ============================================================
# 12. FILE HASH
# ============================================================

def calculate_file_hash(file_bytes):

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# 13. CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(
    uploaded_file,
    embedding_model,
):

    # --------------------------------------------------------
    # Read PDF bytes
    # --------------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    file_hash = calculate_file_hash(
        file_bytes
    )

    # --------------------------------------------------------
    # Save uploaded PDF
    # --------------------------------------------------------

    pdf_path = (
        UPLOAD_DIR /
        uploaded_file.name
    )

    with open(
        pdf_path,
        "wb",
    ) as file:

        file.write(
            file_bytes
        )

    # --------------------------------------------------------
    # Unique ChromaDB directory
    # --------------------------------------------------------

    document_db_path = (
        CHROMA_BASE_PATH /
        file_hash
    )

    document_db_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    loader = PyPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    if not documents:

        raise ValueError(
            "The PDF contains no readable text."
        )

    # --------------------------------------------------------
    # Split PDF into chunks
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:

        raise ValueError(
            "Could not create text chunks."
        )

    # --------------------------------------------------------
    # ADD IMPORTANT METADATA
    #
    # PyPDFLoader gives page numbers starting from 0.
    #
    # Example:
    #
    # PDF page 1 -> page = 0
    # PDF page 2 -> page = 1
    #
    # We convert them to:
    #
    # PDF page 1 -> page_number = 1
    # PDF page 2 -> page_number = 2
    # --------------------------------------------------------

    for chunk in chunks:

        pdf_page = chunk.metadata.get(
            "page"
        )

        if pdf_page is not None:

            chunk.metadata[
                "page_number"
            ] = int(pdf_page) + 1

        else:

            chunk.metadata[
                "page_number"
            ] = "Unknown"

        chunk.metadata[
            "document_name"
        ] = uploaded_file.name

        chunk.metadata[
            "document_hash"
        ] = file_hash

    # --------------------------------------------------------
    # Create ChromaDB
    # --------------------------------------------------------

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(
            document_db_path
        ),
        embedding_function=embedding_model,
    )

    # --------------------------------------------------------
    # Store chunks
    # --------------------------------------------------------

    vectorstore.add_documents(
        documents=chunks
    )

    return (
        vectorstore,
        chunks,
        file_hash,
    )


# ============================================================
# 14. CREATE MULTI-QUERY RETRIEVER
# ============================================================

def create_multi_query_retriever(
    vectorstore,
    llm,
):

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": TOP_K,
        },
    )

    multi_query_retriever = (
        MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            include_original=True,
        )
    )

    return multi_query_retriever


# ============================================================
# 15. GET UNIQUE SOURCE PAGES
# ============================================================

def get_source_pages(
    documents
):

    sources = []

    seen = set()

    for document in documents:

        page_number = document.metadata.get(
            "page_number"
        )

        document_name = document.metadata.get(
            "document_name",
            "Uploaded PDF",
        )

        if page_number == "Unknown":

            continue

        source_key = (
            document_name,
            page_number,
        )

        if source_key not in seen:

            seen.add(
                source_key
            )

            sources.append(
                {
                    "document": document_name,
                    "page": page_number,
                }
            )

    # Sort pages numerically
    sources.sort(
        key=lambda x: (
            x["document"],
            x["page"],
        )
    )

    return sources


# ============================================================
# 16. CREATE PAGE-AWARE CONTEXT
# ============================================================

def create_context(
    documents
):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        page_number = document.metadata.get(
            "page_number",
            "Unknown",
        )

        document_name = document.metadata.get(
            "document_name",
            "Uploaded PDF",
        )

        content = document.page_content

        context_parts.append(
            f"""
[Source {index}]

Document:
{document_name}

PDF Page:
{page_number}

Content:
{content}
""".strip()
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# 17. RAG PROMPT
# ============================================================

def create_rag_prompt():

    return ChatPromptTemplate.from_messages(
        [

            # ------------------------------------------------
            # SYSTEM PROMPT
            # ------------------------------------------------

            (
                "system",
                """
You are an AI assistant for a PDF/book
question-answering system.

Your job is to answer the user's question
ACCORDING TO THE UPLOADED BOOK.

STRICT RULES:

1. Use ONLY the information provided
   in the retrieved context.

2. Do NOT use outside knowledge.

3. Do NOT make up facts.

4. Do NOT assume information that is not
   present in the context.

5. Carefully compare all retrieved sources
   before answering.

6. Give a clear, accurate and useful answer.

7. If the context contains an explanation,
   explain it according to the book.

8. If the context contains an example,
   include the example when it helps answer
   the question.

9. If the answer is not available in the
   retrieved context, say exactly:

"I could not find the answer in the uploaded document."

10. The context contains the actual PDF page
    number for every source.

11. At the end of your answer, provide the
    relevant source pages.

12. Use this format:

📖 Source:
Document Name — Page X

13. If multiple pages are used:

📖 Sources:
Document Name — Pages X, Y, Z

14. NEVER invent a page number.

15. ONLY mention page numbers that appear
    in the context.

16. Do not answer from general knowledge
    when the book does not contain the answer.

17. The uploaded book is the source of truth.

18. If the user's question is ambiguous,
    answer using the interpretation that is
    best supported by the retrieved book content.

"""
            ),

            # ------------------------------------------------
            # HUMAN QUESTION
            # ------------------------------------------------

            (
                "human",
                """
Here is the retrieved content from the
uploaded PDF/book:

==================================================
CONTEXT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}

==================================================

Answer the question strictly according
to the uploaded book.
"""
            ),
        ]
    )


# ============================================================
# 18. PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    current_file_bytes = (
        uploaded_file.getvalue()
    )

    current_hash = calculate_file_hash(
        current_file_bytes
    )

    # --------------------------------------------------------
    # Process only when a NEW PDF is uploaded
    # --------------------------------------------------------

    if (
        st.session_state.document_hash
        != current_hash
    ):

        # ----------------------------------------------------
        # Check API key
        # ----------------------------------------------------

        if not NVIDIA_API_KEY:

            st.error(
                """
                ❌ NVIDIA_API_KEY is missing.

                Add the following to your .env file:

                NVIDIA_API_KEY=your_api_key

                Then restart Streamlit.
                """
            )

            st.stop()

        # ----------------------------------------------------
        # Load embeddings
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Loading embedding model..."
        ):

            embedding_model = (
                load_embedding_model()
            )

        # ----------------------------------------------------
        # Create vector database
        # ----------------------------------------------------

        with st.spinner(
            "📖 Reading PDF and creating vector database..."
        ):

            try:

                (
                    vectorstore,
                    chunks,
                    file_hash,
                ) = create_vector_database(
                    uploaded_file,
                    embedding_model,
                )

            except Exception as error:

                st.error(
                    f"""
                    ❌ Could not process PDF.

                    Error:

                    {error}
                    """
                )

                st.stop()

        # ----------------------------------------------------
        # Load LLM
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Loading NVIDIA Nemotron..."
        ):

            try:

                llm = load_llm(
                    NVIDIA_API_KEY
                )

            except Exception as error:

                st.error(
                    f"""
                    ❌ Could not load NVIDIA Nemotron.

                    Error:

                    {error}
                    """
                )

                st.stop()

        # ----------------------------------------------------
        # Create retriever
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Creating Multi-Query Retriever..."
        ):

            try:

                retriever = (
                    create_multi_query_retriever(
                        vectorstore,
                        llm,
                    )
                )

            except Exception as error:

                st.error(
                    f"""
                    ❌ Could not create retriever.

                    Error:

                    {error}
                    """
                )

                st.stop()

        # ----------------------------------------------------
        # Save in session state
        # ----------------------------------------------------

        st.session_state.vectorstore = (
            vectorstore
        )

        st.session_state.retriever = (
            retriever
        )

        st.session_state.document_name = (
            uploaded_file.name
        )

        st.session_state.document_hash = (
            file_hash
        )

        st.session_state.document_count = (
            len(chunks)
        )

        st.session_state.messages = []

        st.success(
            f"""
            ✅ **{uploaded_file.name}**
            uploaded successfully!

            📦 Created **{len(chunks)} chunks**

            🗄️ ChromaDB created successfully.

            📄 PDF page numbers preserved.
            """
        )


# ============================================================
# 19. NO DOCUMENT MESSAGE
# ============================================================

if st.session_state.retriever is None:

    st.info(
        """
        ### 👋 Welcome to AI Book RAG

        Start by uploading a PDF book from the
        sidebar.

        The application will automatically:

        **1.** Read the PDF

        **2.** Extract PDF pages

        **3.** Split the book into chunks

        **4.** Generate embeddings

        **5.** Store embeddings in ChromaDB

        **6.** Create a Multi-Query Retriever

        **7.** Search relevant pages

        **8.** Generate an answer using NVIDIA Nemotron

        **9.** Show the relevant PDF page numbers
        """
    )

    st.stop()


# ============================================================
# 20. CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 21. CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask a question about your uploaded book..."
)


# ============================================================
# 22. PROCESS USER QUESTION
# ============================================================

if user_query:

    # --------------------------------------------------------
    # Save user question
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    # --------------------------------------------------------
    # Display question
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_query
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Searching the book..."
        ):

            try:

                # ============================================
                # STEP 1 — SEARCH PDF
                # ============================================

                retrieved_docs = (
                    st.session_state
                    .retriever
                    .invoke(
                        user_query
                    )
                )

                # ============================================
                # STEP 2 — CHECK RESULTS
                # ============================================

                if not retrieved_docs:

                    answer = (
                        "I could not find the answer "
                        "in the uploaded document."
                    )

                    st.markdown(
                        answer
                    )

                else:

                    # ========================================
                    # STEP 3 — CREATE PAGE-AWARE CONTEXT
                    # ========================================

                    combined_context = (
                        create_context(
                            retrieved_docs
                        )
                    )

                    # ========================================
                    # STEP 4 — CREATE PROMPT
                    # ========================================

                    rag_prompt = (
                        create_rag_prompt()
                    )

                    formatted_prompt = (
                        rag_prompt.format_prompt(
                            context=combined_context,
                            question=user_query,
                        )
                    )

                    # ========================================
                    # STEP 5 — CALL NEMOTRON
                    # ========================================

                    llm = load_llm(
                        NVIDIA_API_KEY
                    )

                    response = llm.invoke(
                        formatted_prompt
                    )

                    answer = response.content

                    # ========================================
                    # STEP 6 — DISPLAY ANSWER
                    # ========================================

                    st.markdown(
                        "### 🤖 Answer"
                    )

                    st.markdown(
                        answer
                    )

                    # ========================================
                    # STEP 7 — GET ACTUAL SOURCE PAGES
                    # ========================================

                    source_pages = (
                        get_source_pages(
                            retrieved_docs
                        )
                    )

                    # ========================================
                    # STEP 8 — DISPLAY PAGE NUMBERS
                    # ========================================

                    if source_pages:

                        st.markdown(
                            "### 📖 Retrieved PDF Pages"
                        )

                        for source in source_pages:

                            st.markdown(
                                f"""
                                📄 **{source['document']}**
                                — **Page {source['page']}**
                                """
                            )

                    # ========================================
                    # STEP 9 — SHOW RETRIEVED CONTENT
                    # ========================================

                    with st.expander(
                        "🔍 View retrieved pages and text"
                    ):

                        st.write(
                            f"Retrieved "
                            f"**{len(retrieved_docs)}** "
                            f"relevant chunks."
                        )

                        for index, doc in enumerate(
                            retrieved_docs,
                            start=1,
                        ):

                            page_number = (
                                doc.metadata.get(
                                    "page_number",
                                    "Unknown",
                                )
                            )

                            document_name = (
                                doc.metadata.get(
                                    "document_name",
                                    "Uploaded PDF",
                                )
                            )

                            st.markdown(
                                f"""
                                ### Source {index}

                                **Document:** {document_name}

                                **PDF Page:** {page_number}
                                """
                            )

                            st.write(
                                doc.page_content
                            )

                            st.markdown(
                                "---"
                            )

            # ------------------------------------------------
            # ERROR HANDLING
            # ------------------------------------------------

            except Exception as error:

                answer = (
                    "An error occurred while "
                    "processing your question."
                )

                st.error(
                    f"""
                    {answer}

                    **Error:**

                    `{error}`
                    """
                )

    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )