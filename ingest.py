"""Ingest government scheme PDFs from data/schemes/ into a local ChromaDB store.

Usage: python ingest.py
"""
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "schemes"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "schemes"
EMBED_MODEL = os.getenv("GOVASSIST_EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def extract_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "page": index})
    if not pages:
        raise ValueError(f"'{pdf_path.name}' has no extractable text (scanned/image PDF?).")
    return pages


def make_chunks(pdf_name: str, pages: list[dict]) -> tuple[list, list, list]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    scheme_name = Path(pdf_name).stem.replace("_", " ").replace("-", " ").title()
    texts, metadatas, ids = [], [], []
    for page in pages:
        for piece_no, piece in enumerate(splitter.split_text(page["text"])):
            texts.append(piece)
            metadatas.append({"source": pdf_name, "page": page["page"], "scheme": scheme_name})
            ids.append(f"{pdf_name}:p{page['page']}:c{piece_no}")
    return texts, metadatas, ids


def get_db() -> Chroma:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=str(DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest_pdfs(pdfs: list[Path]) -> list[tuple[str, int]]:
    """Ingest the given PDFs into the store, replacing any earlier versions of them."""
    db = get_db()
    results = []
    for pdf in pdfs:
        texts, metadatas, ids = make_chunks(pdf.name, extract_pages(pdf))
        db._collection.delete(where={"source": pdf.name})
        db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        results.append((pdf.name, len(texts)))
    return results


def main() -> None:
    pdfs = sorted(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR}. Add official scheme PDFs there (see data/schemes/README.md)."
        )

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    for name, chunks in ingest_pdfs(pdfs):
        print(f"  {name}: {chunks} chunks")
    print(f"Done. {len(pdfs)} document(s) stored in {DB_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
