import os
import tempfile
import hashlib
import math

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="PDF Chatbot | RAG",
    page_icon="📄",
    layout="centered",
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def get_dynamic_params(total_chars: int):
    """Pick chunk size / overlap based on combined document length."""
    if total_chars < 5_000:
        chunk_size, chunk_overlap = 500, 100
    elif total_chars < 20_000:
        chunk_size, chunk_overlap = 800, 150
    elif total_chars < 60_000:
        chunk_size, chunk_overlap = 1000, 200
    elif total_chars < 150_000:
        chunk_size, chunk_overlap = 1200, 250
    else:
        chunk_size, chunk_overlap = 1500, 300
    return chunk_size, chunk_overlap


def files_hash(files) -> str:
    h = hashlib.md5()
    for f in files:
        h.update(f.name.encode())
        h.update(f.getvalue())
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_vector_store(_files_data: list, group_id: str):
    """
    _files_data: list of (filename, file_bytes)
    Loads every PDF, tags each chunk with its source filename, embeds all
    chunks into one FAISS store. Cached by group_id (hash of all files) so
    re-uploading the same set doesn't re-embed.
    """
    all_docs = []
    per_file_stats = {}
    total_chars = 0
    tmp_paths = []

    try:
        for filename, file_bytes in _files_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_paths.append(tmp.name)

            loader = PyPDFLoader(tmp.name)
            pages = loader.load()
            for p in pages:
                p.metadata["source"] = filename  # tag every page with its file
            chars = sum(len(p.page_content) for p in pages)
            total_chars += chars
            per_file_stats[filename] = {"pages": len(pages), "chars": chars}
            all_docs.extend(pages)

        chunk_size, chunk_overlap = get_dynamic_params(total_chars)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(all_docs)

        embedding_model = get_embedding_model()
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)

        # base k scales with doc length, but we also guarantee per-file coverage below
        if total_chars < 20_000:
            base_k = 5
        elif total_chars < 60_000:
            base_k = 6
        elif total_chars < 150_000:
            base_k = 8
        else:
            base_k = 10

        stats = {
            "num_files": len(_files_data),
            "per_file": per_file_stats,
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "base_k": base_k,
            "sources": list(per_file_stats.keys()),
        }
        return vector_store, stats
    finally:
        for p in tmp_paths:
            os.remove(p)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2,
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions using ONLY the "
            "context provided below, extracted from one or more PDFs the user "
            "uploaded. Each chunk is labeled with its source filename and page.\n\n"
            "Rules:\n"
            "- If the answer isn't in the context, say clearly that you don't "
            "have enough information in the documents to answer that.\n"
            "- Never make up facts that aren't in the context.\n"
            "- If multiple documents are provided and the question asks to "
            "compare, match, or evaluate one against another (e.g. a resume "
            "against a job description), explicitly reference both documents: "
            "identify matching points, gaps/missing requirements, and give a "
            "clear verdict (e.g. strong match / partial match / weak match) "
            "with reasoning.\n"
            "- Keep answers concise and well-structured (use bullet points for "
            "comparisons).\n"
            "- Always mention which document/page the info came from.\n\n"
            "Context:\n{context_text}",
        ),
        ("human", "{user_query}"),
    ]
)


def retrieve_balanced(vector_store, sources: list, query: str, base_k: int):
    """
    Retrieves chunks with guaranteed representation from every uploaded file,
    instead of plain top-k (which could get dominated by one long document).
    """
    if len(sources) <= 1:
        return vector_store.similarity_search(query, k=base_k)

    per_source_k = max(2, math.ceil(base_k / len(sources)))
    results = []
    seen = set()
    for src in sources:
        try:
            hits = vector_store.similarity_search(
                query, k=per_source_k, filter={"source": src}
            )
        except Exception:
            hits = []
        for h in hits:
            key = (h.metadata.get("source"), h.metadata.get("page"), h.page_content[:50])
            if key not in seen:
                seen.add(key)
                results.append(h)
    return results


def answer_query(vector_store, llm, sources: list, user_query: str, base_k: int):
    retrieved_chunks = retrieve_balanced(vector_store, sources, user_query, base_k)

    context_parts = []
    citations = set()
    for doc in retrieved_chunks:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_label = f"page {page + 1}" if isinstance(page, int) else "page ?"
        context_parts.append(f"[{src} | {page_label}]\n{doc.page_content}")
        citations.add(f"{src} (p.{page + 1})" if isinstance(page, int) else src)

    context_text = "\n\n".join(context_parts)

    chain = PROMPT | llm
    result = chain.invoke({"user_query": user_query, "context_text": context_text})

    return result.content, sorted(citations)


# ---------------------------------------------------------
# Sidebar - upload + status
# ---------------------------------------------------------
with st.sidebar:
    st.title("📄 Multi-PDF Chatbot")
    st.caption(
        "Upload multiple PDFs (e.g. a resume + a job description) and ask "
        "questions across all of them — including comparisons."
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not set. Add it in Streamlit secrets / .env.", icon="⚠️")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

    if "stats" in st.session_state:
        st.divider()
        st.subheader("Document stats")
        s = st.session_state["stats"]
        for fname, info in s["per_file"].items():
            st.write(f"**{fname}** — {info['pages']} pages")
        st.caption(f"Total chunks indexed: {s['num_chunks']}")

# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------
st.header("Chat with your PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not uploaded_files:
    st.info("👈 Upload one or more PDFs from the sidebar to get started.")
    st.stop()

files_data = [(f.name, f.getvalue()) for f in uploaded_files]
group_id = files_hash(uploaded_files)

# rebuild only if the set of files changed
if st.session_state.get("current_group_id") != group_id:
    with st.spinner(f"Reading and indexing {len(files_data)} PDF(s)..."):
        try:
            vector_store, stats = build_vector_store(files_data, group_id)
        except Exception as e:
            st.error(f"Couldn't process these PDFs: {e}")
            st.stop()
    st.session_state["vector_store"] = vector_store
    st.session_state["stats"] = stats
    st.session_state["current_group_id"] = group_id
    st.session_state["messages"] = []
    st.toast(f"{len(files_data)} PDF(s) indexed! Ask away.", icon="✅")

vector_store = st.session_state["vector_store"]
stats = st.session_state["stats"]

if stats["num_files"] > 1:
    st.caption(
        "📎 Loaded: " + ", ".join(stats["sources"])
        + " — try asking things like *'How well does the resume match the job description?'*"
    )

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.caption("📌 Sources: " + ", ".join(msg["citations"]))

user_query = st.chat_input("Ask something about your PDF(s)...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not os.environ.get("GROQ_API_KEY"):
            answer = "⚠️ GROQ_API_KEY isn't configured, so I can't call the model. Please set it in secrets."
            citations = []
            st.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm()
                    answer, citations = answer_query(
                        vector_store, llm, stats["sources"], user_query, stats["base_k"]
                    )
                except Exception as e:
                    answer = f"Something went wrong while answering: {e}"
                    citations = []
            st.markdown(answer)
            if citations:
                st.caption("📌 Sources: " + ", ".join(citations))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )