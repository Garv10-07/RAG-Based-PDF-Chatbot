# 📄 RAG PDF Chatbot

Upload any PDF and ask questions about it. Answers are generated only from the
document's content (Retrieval-Augmented Generation), with page-number sources
shown for every answer.

**Stack:** Streamlit · LangChain · FAISS · HuggingFace sentence-embeddings · Groq (Llama 3.3 70B)

## Features
- Upload any PDF (no hardcoded file)
- Dynamic chunking based on document length
- FAISS vector search, cached per-file so re-uploads are instant
- Groq-hosted LLM — fast, free-tier, reliable
- Chat-style UI with history and per-answer page citations
- Clear error handling if the API key is missing or a file fails to load

## Run locally

```bash
git clone <your-repo-url>
cd pdfbot
pip install -r requirements.txt
cp .env.example .env   # then paste your Groq API key into .env
streamlit run app.py
```

Get a free Groq API key at https://console.groq.com/keys

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → **New app** → pick your repo/branch → set
   main file path to `app.py`.
3. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Deploy. First load will take a bit longer (installing `sentence-transformers`
   and downloading the embedding model) — subsequent loads are fast.

## Notes / limitations
- Embeddings run locally (CPU) via `sentence-transformers`, so no extra API key
  is needed for that part.
- One PDF is indexed per session; uploading a new one re-indexes automatically.
- Free Streamlit Cloud has ~1GB RAM — fine for this app, but very large PDFs
  (500+ pages) may be slow to embed on first upload.
