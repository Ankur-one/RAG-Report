import arxiv
from langchain_core.documents import Document


# ============================================================
# ARXIV DOCUMENT LOADER
# ============================================================

QUERY = "1706.03762"
MAX_RESULTS = 2


def load_arxiv_documents(query: str, max_results: int = 2):
    """
    Search and load papers from arXiv.

    Parameters:
        query: arXiv ID, title, author, or search query.
        max_results: Maximum number of papers to retrieve.

    Returns:
        List of LangChain Document objects.
    """

    print("=" * 70)
    print("ARXIV DOCUMENT LOADER")
    print("=" * 70)

    print(f"Query       : {query}")
    print(f"Max results : {max_results}")
    print()

    # --------------------------------------------------------
    # Create arXiv client
    # --------------------------------------------------------

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3.0,
        num_retries=3
    )

    # --------------------------------------------------------
    # Create search
    # --------------------------------------------------------

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    documents = []

    # --------------------------------------------------------
    # Fetch results
    # --------------------------------------------------------

    try:
        results = client.results(search)

        for index, result in enumerate(results, start=1):

            print("-" * 70)
            print(f"DOCUMENT {index}")
            print("-" * 70)

            # ------------------------------------------------
            # Extract paper information
            # ------------------------------------------------

            title = result.title
            summary = result.summary

            authors = [
                author.name
                for author in result.authors
            ]

            published = result.published
            updated = result.updated
            paper_url = result.entry_id
            pdf_url = result.pdf_url

            # ------------------------------------------------
            # Create document content
            # ------------------------------------------------

            content = f"""
Title:
{title}

Authors:
{", ".join(authors)}

Published:
{published}

Updated:
{updated}

Abstract:
{summary}

Paper URL:
{paper_url}

PDF URL:
{pdf_url}
""".strip()

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {
                "source": "arxiv",
                "title": title,
                "authors": ", ".join(authors),
                "published": str(published),
                "updated": str(updated),
                "paper_url": paper_url,
                "pdf_url": pdf_url,
                "arxiv_id": result.get_short_id(),
                "primary_category": result.primary_category,
                "categories": ", ".join(result.categories),
            }

            # ------------------------------------------------
            # LangChain Document
            # ------------------------------------------------

            document = Document(
                page_content=content,
                metadata=metadata
            )

            documents.append(document)

            # ------------------------------------------------
            # Display result
            # ------------------------------------------------

            print(f"Title      : {title}")
            print(f"Authors    : {', '.join(authors)}")
            print(f"arXiv ID   : {result.get_short_id()}")
            print(f"Category   : {result.primary_category}")
            print(f"Paper URL  : {paper_url}")

            print("\nAbstract:")
            print(summary[:1000])

            print()

    except Exception as e:
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"Could not load arXiv papers.")
        print(f"Error: {e}")
        print()
        raise

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING COMPLETED")
    print("=" * 70)

    print(f"Documents loaded: {len(documents)}")

    return documents


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    docs = load_arxiv_documents(
        query=QUERY,
        max_results=MAX_RESULTS
    )

    print()
    print("=" * 70)
    print("LANGCHAIN DOCUMENTS")
    print("=" * 70)

    for i, doc in enumerate(docs, start=1):

        print()
        print(f"Document {i}")
        print("-" * 70)

        print("Page Content:")
        print(doc.page_content[:2000])

        print()
        print("Metadata:")
        print(doc.metadata)

        print("-" * 70)
