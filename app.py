import os
import tempfile
import hashlib

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="PDF Chatbot | RAG",
    page_icon="📄",
    layout="centered",
)

# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------
def get_dynamic_params(document):
    """Pick chunk size / overlap / k based on document length."""
    total_chars = sum(len(page.page_content) for page in document)
    num_pages = len(document)

    if total_chars < 5_000:
        chunk_size, chunk_overlap, k = 500, 100, 4
    elif total_chars < 20_000:
        chunk_size, chunk_overlap, k = 800, 150, 5
    elif total_chars < 60_000:
        chunk_size, chunk_overlap, k = 1000, 200, 6
    elif total_chars < 150_000:
        chunk_size, chunk_overlap, k = 1200, 250, 8
    else:
        chunk_size, chunk_overlap, k = 1500, 300, 10

    return chunk_size, chunk_overlap, k, total_chars, num_pages


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_vector_store(_file_bytes: bytes, file_id: str):
    """Loads a PDF, splits it, embeds it and returns a FAISS store + stats.
    Cached by file_id (hash of file bytes) so re-uploading the same PDF
    doesn't re-embed it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        document = loader.load()

        chunk_size, chunk_overlap, k, total_chars, num_pages = get_dynamic_params(document)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(document)

        embedding_model = get_embedding_model()
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)

        stats = {
            "num_pages": num_pages,
            "total_chars": total_chars,
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "k": k,
        }
        return vector_store, stats
    finally:
        os.remove(tmp_path)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2,
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using ONLY the "
            "context provided below, extracted from a PDF the user uploaded.\n\n"
            "Rules:\n"
            "- If the answer isn't in the context, say clearly that you don't "
            "have enough information in the document to answer that.\n"
            "- Never make up facts that aren't in the context.\n"
            "- Keep answers concise and to the point.\n"
            "- When useful, mention which page(s) the info came from.\n\n"
            "Context:\n{context_text}",
        ),
        ("human", "{user_query}"),
    ]
)


def answer_query(vector_store, llm, user_query: str, k: int):
    retrieved_chunks = vector_store.similarity_search(user_query, k=k)
    context_text = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?') + 1}] {doc.page_content}"
        if isinstance(doc.metadata.get("page"), int)
        else doc.page_content
        for doc in retrieved_chunks
    )
    chain = PROMPT | llm
    result = chain.invoke({"user_query": user_query, "context_text": context_text})

    pages = sorted(
        {
            doc.metadata.get("page") + 1
            for doc in retrieved_chunks
            if isinstance(doc.metadata.get("page"), int)
        }
    )
    return result.content, pages


# ---------------------------------------------------------
# Sidebar - upload + status
# ---------------------------------------------------------
with st.sidebar:
    st.title("📄 PDF Chatbot")
    st.caption("RAG-based Q&A over your PDF, powered by Groq + FAISS.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not set. Add it in Streamlit secrets / .env.", icon="⚠️")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

    if "stats" in st.session_state:
        st.divider()
        st.subheader("Document stats")
        s = st.session_state["stats"]
        st.write(f"**Pages:** {s['num_pages']}")
        st.write(f"**Chunks:** {s['num_chunks']}")
        st.write(f"**Chunk size:** {s['chunk_size']} (overlap {s['chunk_overlap']})")

# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------
st.header("Chat with your PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_id = file_hash(file_bytes)

# rebuild only if a new file was uploaded
if st.session_state.get("current_file_id") != file_id:
    with st.spinner("Reading and indexing your PDF..."):
        try:
            vector_store, stats = build_vector_store(file_bytes, file_id)
        except Exception as e:
            st.error(f"Couldn't process this PDF: {e}")
            st.stop()
    st.session_state["vector_store"] = vector_store
    st.session_state["stats"] = stats
    st.session_state["current_file_id"] = file_id
    st.session_state["messages"] = []
    st.toast("PDF indexed! Ask away.", icon="✅")

vector_store = st.session_state["vector_store"]
stats = st.session_state["stats"]

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.caption(f"📌 Sources: page(s) {', '.join(map(str, msg['pages']))}")

user_query = st.chat_input("Ask something about your PDF...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not os.environ.get("GROQ_API_KEY"):
            answer = "⚠️ GROQ_API_KEY isn't configured, so I can't call the model. Please set it in secrets."
            pages = []
            st.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm()
                    answer, pages = answer_query(vector_store, llm, user_query, stats["k"])
                except Exception as e:
                    answer = f"Something went wrong while answering: {e}"
                    pages = []
            st.markdown(answer)
            if pages:
                st.caption(f"📌 Sources: page(s) {', '.join(map(str, pages))}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "pages": pages})
