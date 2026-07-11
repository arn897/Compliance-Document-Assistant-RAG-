import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

st.set_page_config(page_title="Compliance Document Assistant", layout="wide")
st.title("📄 Compliance Document Assistant (RAG)")
st.caption("Upload a financial filing and ask questions — answers are cited or refused, never fabricated.")

os.environ["HF_TOKEN"] = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

uploaded_file = st.file_uploader("Upload a PDF (10-K, annual report, compliance policy)", type="pdf")

@st.cache_resource
def build_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        provider="featherless-ai",
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    def format_docs(docs):
        return "\n\n".join(f"[Source {i+1}, Page {d.metadata.get('page', 'unknown')}]: {d.page_content}" for i, d in enumerate(docs))

    prompt = ChatPromptTemplate.from_template("""
You are a financial document assistant. Answer the question using ONLY the context below.

Rules:
- If the exact line item asked about is not explicitly present in the context, say "I don't have enough information in this document to answer that" — do NOT substitute a similar-sounding financial metric.
- Do not conflate different financial line items.
- Cite the exact Source number and Page number your answer comes from, e.g. (Source 2, Page 47).

Context:
{context}

Question: {question}

Answer:
""")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Processing document... this may take a minute for large PDFs"):
        rag_chain, retriever = build_chain("temp.pdf")
    st.success("Document ready. Ask a question below.")

    question = st.text_input("Your question:")
    if question:
        with st.spinner("Thinking..."):
            answer = None
            for attempt in range(3):
                try:
                    answer = rag_chain.invoke(question)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(3)
                    else:
                        answer = f"The model is currently busy. Please try again in a moment."
        st.markdown(f"**Answer:** {answer}")

        with st.expander("🔍 View retrieved source chunks"):
            for i, doc in enumerate(retriever.invoke(question)):
                st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'unknown')})**")
                st.text(doc.page_content[:400])
                st.divider()
