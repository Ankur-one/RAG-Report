import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError(
        "NVIDIA_API_KEY not found in .env file."
    )


# ============================================================
# 2. NVIDIA NEMOTRON MODEL
# ============================================================

model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.2,
    max_tokens=1000,
)


# ============================================================
# 3. DOCUMENT LOADER
# ============================================================

def load_document(file_path):
    """
    Load PDF, DOCX or TXT file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )

    else:
        raise ValueError(
            "Unsupported file format. "
            "Use PDF, DOCX or TXT."
        )

    return loader.load()


# ============================================================
# 4. SPLIT DOCUMENT
# ============================================================

def split_document(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ============================================================
# 5. SUMMARIZE CHUNKS
# ============================================================

def summarize_chunks(chunks):

    summaries = []

    for i, chunk in enumerate(chunks):

        print(
            f"Summarizing chunk {i + 1}/{len(chunks)}..."
        )

        prompt = f"""
You are an expert document summarizer.

Summarize the following document section clearly.

Requirements:
- Keep the important information.
- Remove unnecessary repetition.
- Use simple and clear language.
- Preserve important facts, concepts and terminology.
- Do not invent information.
- Use bullet points where appropriate.

DOCUMENT SECTION:

{chunk.page_content}

SUMMARY:
"""

        response = model.invoke(prompt)

        summaries.append(response.content)

    return summaries


# ============================================================
# 6. COMBINE SUMMARIES
# ============================================================

def create_final_summary(summaries):

    combined_summary = "\n\n".join(summaries)

    prompt = f"""
You are an expert academic document summarizer.

Create one final summary from the section summaries below.

Requirements:
- Give a clear overall summary.
- Include the main topics and important concepts.
- Remove duplicate information.
- Keep important facts.
- Use headings and bullet points.
- Do not add information that is not present.
- Make the result useful for a student.

SECTION SUMMARIES:

{combined_summary}

FINAL SUMMARY:
"""

    response = model.invoke(prompt)

    return response.content


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    file_path = input(
        "Enter PDF/DOCX/TXT file path: "
    ).strip()

    print("\nLoading document...")

    documents = load_document(file_path)

    print(
        f"Loaded {len(documents)} document pages/sections."
    )

    print("\nSplitting document...")

    chunks = split_document(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    print("\nGenerating summaries...\n")

    summaries = summarize_chunks(chunks)

    print("\nCreating final summary...")

    final_summary = create_final_summary(
        summaries
    )

    print("\n")
    print("=" * 70)
    print("FINAL DOCUMENT SUMMARY")
    print("=" * 70)

    print(final_summary)

    print("=" * 70)