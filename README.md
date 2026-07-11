# 📄 Compliance Document Assistant — RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions about financial and compliance documents (10-Ks, annual reports, regulatory filings) with **cited, grounded answers** — and explicitly refuses to answer when the information isn't present in the source document, rather than hallucinating figures.

**🔗 Live Demo:** [[your-streamlit-url-here](https://e6hf3igpqcyuldeizpjdzx.streamlit.app/)]
**🎥 Video Demo:** [your-video-link-here]

---

## Why this project

Financial and compliance teams (AML/KYC, audit, risk) need to extract precise information from lengthy regulatory documents — but LLMs are prone to hallucinating numbers, which is unacceptable in a compliance context. This project was built to explore how far prompt design and retrieval tuning can go toward a **trustworthy, non-hallucinating** document Q&A system, using entirely free/open-source tooling.

## Features

- 📤 Upload any PDF (10-K, annual report, compliance policy document)
- 🔍 Semantic search over document chunks using FAISS vector store
- 🤖 Grounded LLM answers — refuses to answer if the information isn't explicitly in the document
- 📌 Page-level source citations for every answer, so claims are independently verifiable
- 🕵️ Expandable "view source chunks" panel — see exactly what context the model used
- ♻️ Retry logic for third-party inference API instability

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain (LCEL) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| LLM | Qwen2.5-7B-Instruct (via HuggingFace Inference Providers) |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

## Architecture
PDF Upload
│
▼
PyPDFLoader → RecursiveCharacterTextSplitter (800 chars, 100 overlap)
│
▼
HuggingFace Embeddings → FAISS Vector Store
│
▼
User Question → Retriever (top-k=8 chunks)
│
▼
Strict Grounding Prompt + Retrieved Context → Qwen2.5-7B-Instruct
│
▼
Cited Answer (Source #, Page #) — or explicit refusal if unsupported

## Evaluation

Manually evaluated against 5 ground-truth questions derived from a 233-page 10-K filing, verified against the source document:

| Metric | Result |
|---|---|
| Factual accuracy | 5/5 (100%) |
| Hallucination rate | 0/5 (0%) |
| Correct refusal on out-of-scope questions | 1/1 |

**Example result:**

> **Q:** What were the general and administrative expenses for 2024 compared to 2023?
> **A:** The general and administrative expenses for 2024 were $48,672,000, compared to $49,565,000 for 2023 — a decrease of $893,000. *(Source 1, Page 74)*

> **Q:** What was the total revenue this year?
> **A:** I don't have enough information in this document to answer that.
> *(Correctly refused — this document does not report a standard "revenue" line item.)*

During development, an early version of the prompt caused the model to conflate "net cash from operating activities" with "net increase in net assets from operations" — two distinct financial line items. This was caught during manual evaluation and fixed by tightening the prompt to explicitly prohibit conflating similar-sounding metrics, after which the model correctly refused rather than guessing.

## Design Decisions

- **Chunk size (800/100 overlap):** Small enough for precise retrieval of specific figures, large enough to preserve full sentences/table rows containing financial data.
- **Strict grounding prompt:** The system prompt explicitly instructs the model to refuse rather than substitute a similar metric — critical for compliance use cases where a wrong number is worse than no answer.
- **Manual evaluation over automated frameworks (e.g. RAGAS):** RAGAS was initially attempted but had unresolved dependency conflicts with current LangChain versions; a small, manually-verified ground-truth test set was used instead as a more reliable and transparent alternative.

## Running Locally

```bash
git clone https://github.com/your-username/compliance-document-rag.git
cd compliance-document-rag
pip install -r requirements.txt
```

Add your HuggingFace token as an environment variable or in `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "your_huggingface_token"
```

Run:
```bash
streamlit run app.py
```

## Limitations & Future Work

- No multi-turn conversation memory (each question is independent)
- Single-document only (no cross-document comparison)
- Free-tier inference API occasionally rate-limited (retry logic included)
- Could be extended with a cross-encoder re-ranker for improved retrieval precision on longer documents

## Author

**Arnav Rana**
[GitHub](https://github.com/arn897) · [LinkedIn](https://linkedin.com/in/arnav-rana-16b45b258) · rarnav10@gmail.com
