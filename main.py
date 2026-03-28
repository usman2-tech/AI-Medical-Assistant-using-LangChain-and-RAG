import os
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Medical Assistant")
st.caption("Upload one or more PDFs and ask questions about them.")


# -------------------------------------------------
# LOAD ENV
# -------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it to your .env file first.")
    st.stop()


# -------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def save_uploaded_file_temporarily(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def load_pdf_documents(uploaded_files) -> List[Document]:
    all_docs = []
    temp_paths = []

    try:
        for uploaded_file in uploaded_files:
            temp_path = save_uploaded_file_temporarily(uploaded_file)
            temp_paths.append(temp_path)

            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = uploaded_file.name

            all_docs.extend(docs)

        return all_docs

    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception:
                pass


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def format_chat_history(messages: List[dict]) -> str:
    if not messages:
        return "No previous conversation."

    history_lines = []
    for msg in messages[-10:]:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        history_lines.append(f"{role}: {content}")

    return "\n".join(history_lines)


def build_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI PDF assistant.

Answer the user's question using only the provided context.

Rules:
1. If the answer is clearly present in the context, answer naturally and clearly.
2. If the answer is not present, say: "I couldn't find that in the uploaded PDFs."
3. If possible, mention source file names or page numbers if the context makes them clear.
4. Do not make up facts.

Chat History:
{chat_history}

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough(),
            "chat_history": RunnableLambda(
                lambda _: format_chat_history(st.session_state.get("messages", []))
            ),
        }
        | document_chain
    )

    return rag_chain


def process_pdfs(uploaded_files):
    documents = load_pdf_documents(uploaded_files)

    if not documents:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    chunks = split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    rag_chain = build_rag_chain(retriever)

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.session_state.rag_chain = rag_chain
    st.session_state.pdf_ready = True
    st.session_state.processed_files = [file.name for file in uploaded_files]


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                try:
                    process_pdfs(uploaded_files)
                    st.success("PDFs processed successfully.")
                except Exception as e:
                    st.error(f"Error while processing PDFs: {e}")

    st.markdown("---")

    if st.session_state.processed_files:
        st.subheader("Processed Files")
        for file_name in st.session_state.processed_files:
            st.write(f"• {file_name}")

    st.markdown("---")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset App", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.rag_chain = None
        st.session_state.pdf_ready = False
        st.session_state.processed_files = []
        st.rerun()


# -------------------------------------------------
# CHAT HISTORY DISPLAY
# -------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------------------------------
# CHAT INPUT
# -------------------------------------------------
user_query = st.chat_input("Ask a question about your PDFs...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not st.session_state.pdf_ready or st.session_state.rag_chain is None:
            response = "Please upload and process your PDF files first."
            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
        else:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(user_query)

                    if not isinstance(response, str):
                        response = str(response)

                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    error_message = f"Error generating answer: {e}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )