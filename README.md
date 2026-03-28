# AI-Medical-Assistant-using-LangChain-and-RAG
AI Medical Assistant using LangChain and RAG


#  AI Medical PDF Assistant (RAG + LangChain Latest)

An **AI-powered Medical Assistant** that allows users to upload medical PDFs (reports, prescriptions, research papers) and ask questions based strictly on the document content using **Retrieval-Augmented Generation (RAG)**.
Built using the **latest architecture of LangChain (LCEL)** and **Google Gemini**, this project demonstrates modern best practices for building production-ready AI systems.

---

##  Features

* 📄 Upload multiple PDF documents
* 🔍 Context-aware retrieval using FAISS
* 💬 Conversational Q&A over documents
* 🧠 Chat history awareness
* ⚠️ Reduces hallucination (answers grounded in PDFs only)
* ⚡ Built using latest LangChain (no deprecated APIs)
* 🌐 Simple UI using Streamlit

---

##  How It Works

1. Upload PDF files
2. PDFs are parsed and split into chunks
3. Chunks are converted into embeddings
4. Stored in a FAISS vector database
5. User query → retrieves relevant chunks
6. LLM generates answer using retrieved context

---

##  Tech Stack

* **LLM:** Google Gemini
* **Framework:** LangChain (LCEL architecture)
* **Vector Database:** FAISS
* **Frontend:** Streamlit
* **Embeddings:** Gemini (`gemini-embedding-001`)
* **Language:** Python

---

##  LangChain (Latest Changes)

This project uses **modern LangChain architecture** instead of deprecated APIs.


### Key Concepts Used:

* Runnable pipelines
* LCEL (LangChain Expression Language)
* Modular chain composition

---

##  Project Structure

```bash
AI-pdf-assistant/
│
├── main.py
├── .env
├── requirements.txt
└── README.md
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-medical-pdf-assistant.git
cd ai-medical-pdf-assistant
```

---

### 2. Create virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate
```

---

### 3. Install dependencies

```bash
pip install -U streamlit python-dotenv pypdf faiss-cpu \
langchain langchain-core langchain-community \
langchain-google-genai langchain-text-splitters
```

---

### 4. Add API Key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

### 5. Run the app

```bash
python -m streamlit run main.py
```

---

##  Example Use Cases

* 📑 Analyze medical reports
* 🧬 Understand research papers
* 💊 Extract prescription details
* 📊 Summarize diagnostic documents






