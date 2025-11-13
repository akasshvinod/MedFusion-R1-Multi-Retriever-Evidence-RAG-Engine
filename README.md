# 🧠 MedFusion-R1: Multi-Retriever Evidence RAG Engine 
**A Production-Ready Medical AI System using DeepSeek R1, ChromaDB, PubMed, Wikipedia, and LangChain v1.x**

**MedFusion-R1 is a professional-grade Medical RAG Engine designed to retrieve, fuse, and reason over multiple medical knowledge sources**

This project implements a **high-accuracy medical RAG system** combining:

- **DeepSeek R1T2 Chimera (via OpenRouter)**
- **ChromaDB vectorstore** for local NIH MedQuAD retrieval
- **Multi-source retrieval (Chroma + Wikipedia API + PubMed)**
- **LLM-driven intelligent source selection**
- **Streaming responses**
- **Memory-aware conversation system**
- **Prompt engineering for medical safety**
- **RunnableParallel/RunnableLambda RAG pipelines**
- **Streamlit frontend**
- **Production-Style Modular Codebase**

This is a **production-grade architecture**, built with clean modular code and industry-standard best practices.

---

## ⚙️ Tech Stack

| Component | Technology |
|----------|------------|
| LLM | DeepSeek R1T2 Chimera (OpenRouter) |
| Vector DB | ChromaDB (with HF Embeddings) |
| Data | NIH MedQuAD Dataset |
| RAG Framework | LangChain v1.x Runnables |
| Tools | Wikipedia API, PubMed API |
| Memory | Custom MemoryManager with summarization |
| Frontend | Streamlit |
| Environment | Conda + Python 3.10 |

---

## 📂 Project Structure

DeepSeek-MCP-Medical-RAG/
│
├── data/ # Raw MedQuAD XML files
│ └── MedQuAD/
│
├── docs/ # Cleaned Q&A text files for Chroma ingestion
│
├── chroma_store/ # Persistent vector embeddings
│
├── src/
│ ├── convert_medquad_xml_to_txt.py # Convert XML → text
│ ├── rag_build_index_chroma.py # Build Chroma vector DB
│ ├── deepseek_api.py # DeepSeek/OpenRouter wrapper
│ ├── rag_agent_mcp.py # Master RAG + MCP pipeline
│ ├── memory_manager.py # Multi-turn memory + summarization
│ ├── prompts.py # Prompt templates
│ └── utils.py # Helpers
│
├── app/
│ └── streamlit_app.py # Frontend interface
│
├── notebooks/
│ ├── 01_Data_Preparation.ipynb
│ ├── 02_Indexing_Chroma.ipynb
│ ├── 03_RAG_Querying.ipynb
│ └── 04_Streamlit_Test.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore


---

## 🧪 Features

### ✔ **Multi-Source Medical RAG**
Runs 3 retrievers in parallel using `RunnableParallel`:

- Local ChromaDB  
- Wikipedia API  
- PubMed research abstracts  

### ✔ **LLM-Driven Smart Source Selection**
DeepSeek analyzes the question and chooses the best sources:

```json
{"chroma": true, "wikipedia": true, "pubmed": false}
