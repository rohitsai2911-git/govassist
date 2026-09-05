import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import ingest

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "schemes"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "schemes"
EMBED_MODEL = os.getenv("GOVASSIST_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("GOVASSIST_LLM_MODEL", "gpt-4o-mini")
TOP_K = 6
MIN_RELEVANCE = 0.25

SYSTEM_PROMPT = """You are GovAssist, an assistant for finding Indian government schemes in uploaded documents.
Follow these rules strictly:
- Answer ONLY from the supplied context passages. Never invent scheme names, eligibility criteria, benefit amounts, deadlines, or application procedures.
- Prefer giving a grounded answer over refusing: if the passages contain relevant information, state what they say, and note any missing details instead of refusing entirely.
- You may connect the user's situation (e.g., their state or category) to eligibility conditions explicitly stated in the passages.
- Only if the passages contain nothing relevant to the question, say exactly: "The available documents do not contain enough information about this."
- Clearly separate what the documents state from reasonable interpretation; never present guesses as facts.
- Keep the answer concise and practical.
- End with a reminder to verify current eligibility and application details through official government sources."""


@st.cache_resource(show_spinner=False)
def get_store():
    try:
        return Chroma(
            persist_directory=str(DB_DIR),
            collection_name=COLLECTION,
            embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        st.error(f"Failed to initialize the document database: {exc}")
        return None


def retrieve(store, question: str) -> list[tuple]:
    results = store.similarity_search_with_relevance_scores(question, k=TOP_K)
    return [(doc, score) for doc, score in results if score >= MIN_RELEVANCE]


def answer_question(question: str, results: list[tuple]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata['source']} | Page {doc.metadata['page']}]\n{doc.page_content}"
        for doc, _ in results
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    return llm.invoke(messages).content


def show_sources(results: list[tuple]) -> None:
    seen, lines = set(), []
    for doc, _ in results:
        key = (doc.metadata["source"], doc.metadata["page"])
        if key not in seen:
            seen.add(key)
            lines.append(f"- `{key[0]}` — Page {key[1]}")
    st.markdown("\n".join(lines))


st.set_page_config(page_title="GovAssist", page_icon="🏛️")
st.title("🏛️ GovAssist")
st.subheader("Find relevant Indian government schemes using AI-powered document retrieval.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Copy `.env.example` to `.env` and add your key.")
    st.stop()

store = get_store()
if store is None:
    st.stop()

pdf_count = len(list(DATA_DIR.glob("*.pdf"))) if DATA_DIR.exists() else 0
if pdf_count:
    st.caption(f"📚 {pdf_count} scheme document(s) loaded from `data/schemes/`.")
else:
    st.caption("⚠️ No PDFs found. Upload below or place government scheme PDFs in `data/schemes/`.")

uploads = st.file_uploader("Add scheme PDFs", type=["pdf"], accept_multiple_files=True)
if uploads and st.button(f"Ingest {len(uploads)} uploaded file(s)"):
    saved = []
    for upload in uploads:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest = DATA_DIR / upload.name
        dest.write_bytes(upload.getvalue())
        saved.append(dest)
    try:
        with st.spinner("Extracting, chunking, and embedding documents..."):
            results = ingest.ingest_pdfs(saved)
        summary = ", ".join(f"{name}: {chunks} chunks" for name, chunks in results)
        st.success(f"Ingested! ({summary})")
        st.cache_resource.clear()
        st.rerun()
    except Exception as exc:
        st.error(f"Ingestion failed: {exc}")

question = st.text_input("Your question", placeholder="e.g. Which schemes support women entrepreneurs?")
ask = st.button("Ask")

if ask:
    if not question.strip():
        st.warning("Please enter a valid question.")
        st.stop()
    with st.spinner("Searching scheme documents..."):
        results = retrieve(store, question)
    if not results:
        st.info("No relevant passages found in the indexed documents. Try rephrasing your "
                "question, or add more scheme PDFs and re-run ingestion.")
        st.stop()
    st.markdown("### Answer")
    with st.spinner("Generating grounded answer..."):
        st.write(answer_question(question, results))
    st.markdown("### Sources")
    show_sources(results)

st.divider()
st.caption(
    "This tool provides information from the uploaded documents and should not be treated as "
    "official government advice. Verify current eligibility and application details with the "
    "relevant government authority."
)
