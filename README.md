# 📚 AI Book RAG Assistant

> **A PDF-based Retrieval-Augmented Generation (RAG) application that allows users to upload a book/document, ask questions, retrieve the most relevant content, and receive answers grounded strictly in the uploaded document with PDF page references.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-orange)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron-76B900?logo=nvidia)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Project Overview

**AI Book RAG Assistant** is a Retrieval-Augmented Generation application designed to answer questions from uploaded PDF books and study materials.

Instead of asking an LLM to answer from its general knowledge, this project first searches the uploaded document for relevant information and then provides that information to the LLM as context.

This approach helps produce answers that are:

* 📖 Grounded in the uploaded document
* 🎯 Relevant to the user's question
* 🔎 Based on retrieved document content
* 📄 Traceable to PDF page numbers
* 🚫 Less dependent on outside/general knowledge

The application is especially useful for **students, researchers, technical documentation, textbooks, and study materials**.

---

# 🎯 Key Features

### 📄 PDF Upload

Upload a PDF book or study document directly through the Streamlit interface.

### 🧠 Automatic Document Processing

The application automatically:

1. Loads the PDF
2. Extracts the text
3. Splits the text into smaller chunks
4. Preserves PDF page numbers
5. Generates vector embeddings
6. Stores embeddings in ChromaDB

### 🔎 Multi-Query Retrieval

The application uses Multi-Query Retrieval to improve document searching.

Instead of relying on only one interpretation of a question, the system generates alternative query representations and searches the vector database for relevant information.

### 🤖 NVIDIA Nemotron

The retrieved document context is passed to NVIDIA Nemotron to generate the final answer.

The model is instructed to answer **only from the retrieved document context**.

### 📖 PDF Page Citations

Retrieved chunks preserve their original PDF page numbers.

For example:

```text
📖 Sources:

Deep Learning.pdf — Page 125
Deep Learning.pdf — Page 126
Deep Learning.pdf — Page 128
```

This makes it easier to verify the answer against the original book.

### 💬 Conversational Interface

The application provides a chat-style interface using Streamlit.

Users can ask multiple questions about the uploaded document.

### 🔐 API Key Protection

API credentials are loaded from `.env` rather than being hard-coded into the application.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────┐
                         │    PDF / Book    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  PyPDFLoader     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Text Chunking    │
                         │ Recursive        │
                         │ Character       │
                         │ Splitter         │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ HuggingFace Embeddings  │
                    │ all-MiniLM-L6-v2        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                         ┌──────────────────┐
                         │    ChromaDB      │
                         │  Vector Database │
                         └────────┬─────────┘
                                  │
                                  │
                         User Question
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   Multi-Query Retriever  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                       Relevant PDF Chunks
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     RAG Prompt           │
                    │ Context + Question       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   NVIDIA Nemotron        │
                    │      LLM                 │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ Final Answer     │
                       │ + Page Sources   │
                       └──────────────────┘
```

---

# 🔄 RAG Workflow

The complete workflow is:

```text
1. Upload PDF
       ↓
2. Extract PDF text
       ↓
3. Split text into chunks
       ↓
4. Preserve page metadata
       ↓
5. Generate embeddings
       ↓
6. Store embeddings in ChromaDB
       ↓
7. User asks a question
       ↓
8. Generate multiple search queries
       ↓
9. Retrieve relevant chunks
       ↓
10. Build context
       ↓
11. Send context + question to Nemotron
       ↓
12. Generate answer
       ↓
13. Display relevant PDF pages
```

---

# 🧩 Technology Stack

| Component              | Technology                     |
| ---------------------- | ------------------------------ |
| Programming Language   | Python                         |
| User Interface         | Streamlit                      |
| LLM                    | NVIDIA Nemotron                |
| LLM API                | NVIDIA API                     |
| RAG Framework          | LangChain                      |
| Embeddings             | Sentence Transformers          |
| Embedding Model        | `all-MiniLM-L6-v2`             |
| Vector Database        | ChromaDB                       |
| PDF Processing         | PyPDFLoader                    |
| Text Splitting         | RecursiveCharacterTextSplitter |
| Retrieval              | Multi-Query Retriever          |
| Environment Management | python-dotenv                  |

---

# 📁 Project Structure

```text
RAG-Report/
│
├── 📄 main.py
├── 📄 create_databases.py
├── 📄 requirements.txt
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 .gitignore
├── 📄 .env.example
│
├── 📁 document loaders/
│   ├── test.py
│   ├── TokenText.py
│   └── summarize.py
│
├── 📁 retrievers/
│   ├── arxiv_loader.py
│   └── multi_query_retriever.py
│
├── 📁 vector_store/
│   └── DB.py
│
├── 📁 src/
│
├── 📁 chroma_db/
│   └── Generated locally
│
└── 📁 uploaded_documents/
    └── Generated locally
```

> `chroma_db/`, `uploaded_documents/`, `.env`, and `.venv/` are excluded from Git using `.gitignore`.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Ankur-one/RAG-Report.git
```

Move into the project:

```bash
cd RAG-Report
```

---

# 🐍 2. Create Virtual Environment

Using Python:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# 📦 3. Install Dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using `uv`:

```powershell
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
```

---

# 🔑 4. Configure NVIDIA API Key

Create a `.env` file in the project root:

```text
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Do **not** commit `.env` to GitHub.

The repository contains:

```text
.env.example
```

as a template.

---

# ▶️ 5. Run the Application

Run:

```powershell
streamlit run main.py
```

Or:

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Open the Streamlit URL shown in the terminal.

Usually:

```text
http://localhost:8501
```

---

# 📖 How to Use

### Step 1 — Upload a PDF

Open the application and upload your book/document from the sidebar.

Example:

```text
Deep Learning.pdf
```

### Step 2 — Wait for Processing

The application automatically creates:

```text
PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
ChromaDB
```

### Step 3 — Ask a Question

Example:

```text
What is backpropagation?
```

### Step 4 — Retrieve Relevant Content

The Multi-Query Retriever searches the document for relevant chunks.

### Step 5 — Generate Answer

NVIDIA Nemotron receives:

```text
Retrieved Context
+
User Question
```

and generates the answer.

### Step 6 — Verify the Source

The application displays:

```text
📖 Retrieved PDF Pages

Deep Learning.pdf — Page 127
Deep Learning.pdf — Page 128
```

You can then verify the answer directly in the book.

---

# 💡 Example

### User Question

```text
What is a convolutional neural network?
```

### RAG Pipeline

```text
Question
   ↓
Multi-Query Retrieval
   ↓
ChromaDB
   ↓
Relevant chunks
   ↓
PDF pages 45, 46, 47
   ↓
NVIDIA Nemotron
```

### Output

```text
A Convolutional Neural Network (CNN) is a
deep learning architecture designed primarily
for processing data with grid-like structures,
such as images.

📖 Sources:
Deep Learning.pdf — Pages 45, 46
```

---

# 🧠 Why RAG?

Traditional LLM:

```text
Question
   ↓
LLM
   ↓
Answer
```

The model may use information from its training data.

RAG:

```text
Question
   ↓
Document Search
   ↓
Relevant Information
   ↓
LLM
   ↓
Answer
```

This allows the answer to be grounded in a specific document.

---

# 🔍 Multi-Query Retrieval

A single user question may not always retrieve the best document chunks.

For example:

```text
User:

Explain Word2Vec.
```

The retriever can consider variations such as:

```text
What is Word2Vec?

How does Word2Vec work?

How are word embeddings generated using Word2Vec?

Explain the Word2Vec framework.
```

These queries are used to improve retrieval coverage.

---

# 📄 Page-Aware Retrieval

Every PDF chunk stores metadata such as:

```text
document_name
document_hash
page_number
```

Example:

```python
{
    "document_name": "Deep Learning.pdf",
    "document_hash": "...",
    "page_number": 127
}
```

Therefore, when a chunk is retrieved, the original PDF page can also be displayed.

---

# 🛡️ Grounded Answering

The RAG prompt instructs the LLM:

```text
Use ONLY the information provided
in the retrieved context.

Do NOT use outside knowledge.

Do NOT invent facts.

Do NOT invent page numbers.
```

If the answer cannot be found:

```text
I could not find the answer in the uploaded document.
```

This helps reduce unsupported answers.

---

# 💰 Cost-Efficient Architecture

The project uses local embeddings:

```text
all-MiniLM-L6-v2
```

Therefore, embeddings do not require a paid OpenAI embedding API.

Architecture:

```text
             Local
        ┌───────────────┐
        │ HuggingFace   │
        │ Embeddings    │
        └───────┬───────┘
                │
                ▼
             ChromaDB
                │
                │
          NVIDIA API
                │
                ▼
        NVIDIA Nemotron
```

This makes the project suitable for learning, prototyping, and academic projects.

---

# 🔐 Security

Sensitive files are excluded from GitHub:

```text
.env
.venv/
chroma_db/
uploaded_documents/
```

Never upload your real API key to GitHub.

If an API key is accidentally exposed, revoke it and generate a new one.

---

# 🚧 Current Limitations

The current version primarily supports:

* PDF documents
* Text-based PDFs
* Single active uploaded document
* Vector similarity retrieval
* Multi-query retrieval
* NVIDIA Nemotron

Scanned/image-only PDFs may require OCR before their text can be retrieved effectively.

---

# 🔮 Future Improvements

Planned improvements include:

* [ ] Multiple PDF support
* [ ] DOCX support
* [ ] TXT and Markdown support
* [ ] OCR for scanned books
* [ ] Hybrid search
* [ ] BM25 + vector search
* [ ] Reranking
* [ ] Better citation generation
* [ ] Clickable PDF page citations
* [ ] Page preview
* [ ] Document comparison
* [ ] Conversation memory
* [ ] RAG evaluation using RAGAS
* [ ] Streaming LLM responses
* [ ] Authentication
* [ ] Cloud deployment
* [ ] Docker support

---

# 📊 Project Learning Outcomes

Through this project, the following concepts are demonstrated:

### Python

* File handling
* Environment variables
* Functions
* Exception handling
* Object-oriented libraries

### LangChain

* Document loaders
* Document splitting
* Embeddings
* Vector stores
* Retrievers
* Prompt templates
* LLM integration

### RAG

* Document ingestion
* Chunking
* Embedding generation
* Vector search
* Context retrieval
* Context-grounded generation

### Vector Database

* ChromaDB
* Persistent vector storage
* Similarity search
* Metadata filtering

### Generative AI

* LLM integration
* Prompt engineering
* Context-aware generation
* Hallucination reduction

### Streamlit

* File upload
* Session state
* Chat interface
* Interactive UI

---

# 🎓 Use Cases

This project can be used for:

* 📚 Student study assistants
* 📖 Textbook question answering
* 🧑‍🎓 Exam preparation
* 🔬 Research paper analysis
* 📄 Technical documentation Q&A
* 🏫 Educational platforms
* 📑 Document summarization
* 🤖 Knowledge assistants

---

# 👨‍💻 Author

## Ankur Kumar

**B.Tech Computer Science & Engineering Student**

Interested in:

* Artificial Intelligence
* Machine Learning
* Data Science
* Generative AI
* Retrieval-Augmented Generation
* Python Development

### Connect with me

* GitHub: [Ankur-one](https://github.com/Ankur-one)
* LinkedIn: [Ankur Kumar](https://www.linkedin.com/in/ankur-kumar-198031287/)
* Kaggle: [Ankur Kumar](https://www.kaggle.com/ankur2005kumar)

---

# ⭐ If You Like This Project

If you find this project useful for learning about **RAG, LangChain, Vector Databases, and Generative AI**, consider giving the repository a ⭐ on GitHub.

---

# 📜 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.

```

**One correction before you paste it:** if your actual LinkedIn/Kaggle URLs differ from the ones above, replace those two links with your exact profile URLs.

For a recruiter-facing GitHub, I would also add **2–3 screenshots of your Streamlit UI** under the “Example” section. That will make the repository look substantially more complete.
```
