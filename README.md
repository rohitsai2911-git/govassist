# GovAssist — Indian Government Scheme Finder

A small RAG application that answers questions about Indian government schemes **only from the documents you provide**, with page-level source citations.

## Problem

India has hundreds of central and state welfare schemes spread across dozens of ministry and state portals. Students, women entrepreneurs, and farmers often don't know which schemes they qualify for or where to look.

## Solution

GovAssist lets you drop official scheme PDFs into a folder, indexes them into a local vector database, and answers natural-language questions using retrieval-augmented generation (RAG). Every answer cites the exact document and page it came from — and the app explicitly says when the documents don't contain enough information, instead of guessing.

## RAG Architecture

```
PDFs (data/schemes/)
  → text extraction (pypdf)
  → chunking (1000 chars / 150 overlap)
  → embeddings (OpenAI text-embedding-3-small)
  → ChromaDB (local persistent store)

User Question
  ↓
Embedding
  ↓
ChromaDB Retrieval (top-4 chunks, relevance-filtered)
  ↓
Relevant Scheme Passages
  ↓
LLM (gpt-4o-mini, grounded system prompt)
  ↓
Grounded Answer + Sources
```

The LLM never sees the full PDFs — only the few retrieved passages. Its system prompt forbids inventing scheme names, eligibility criteria, amounts, or deadlines, and requires an explicit "not enough information" response when the context is insufficient.

## Tech Stack

| Component      | Choice                              |
|----------------|-------------------------------------|
| UI             | Streamlit                           |
| PDF extraction | pypdf                               |
| Chunking       | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings     | OpenAI `text-embedding-3-small`     |
| Vector store   | ChromaDB (local, persisted)         |
| Generation     | OpenAI `gpt-4o-mini`                |
| Secrets        | python-dotenv                       |

## How to Run

```bash
cd govassist
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your OpenAI API key into .env

# 1. Add official scheme PDFs to data/schemes/ (see data/schemes/README.md)
python ingest.py              # extracts, chunks, embeds, stores in chroma_db/

streamlit run app.py
```

## Example Questions

- "What schemes are available for college students?"
- "I am a student from Karnataka. What financial assistance schemes might apply to me?"
- "Which schemes support women entrepreneurs?"
- "What are the eligibility requirements for this scheme?"

## Website

A standalone landing page lives at `website/index.html` (pure HTML/CSS, no build step). Open it directly in a browser or serve it with `python -m http.server -d website`. It explains the pipeline, where to download official scheme PDFs, and the upload rules.

## Limitations

- Answers are only as good as the uploaded PDFs; nothing outside them is used.
- Scanned/image PDFs need OCR before ingestion (not implemented).
- Scheme rules change frequently — always verify with the issuing authority.
- Retrieval quality depends on chunking; very tabular or layout-heavy PDFs may split awkwardly.
- Single-user local MVP: no auth, no multi-document collections, no evaluation harness.
