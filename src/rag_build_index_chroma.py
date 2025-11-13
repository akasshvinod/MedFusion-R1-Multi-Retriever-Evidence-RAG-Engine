"""
rag_build_index_chroma.py
------------------------------------------------------
Builds and persists a ChromaDB vector index from MedQuAD Q&A pairs in TXT.
"""

import os
import torch
from tqdm import tqdm
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document

def get_device(force_cpu=False):
    """
    Returns 'cuda' if available else 'cpu', unless force_cpu=True.
    """
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_medquad_txt(txt_path: str):
    """
    Loads medquad_qa_corpus.txt as a list of LangChain Document objects.
    Assumes format:
      Q: <question>
      A: <answer>
      <blank line>
    Returns: List[Document]
    """
    documents = []
    with open(txt_path, "r", encoding="utf-8") as f:
        entries = f.read().strip().split("\n\n")
    for entry in entries:
        entry = entry.strip()
        if entry:
            documents.append(Document(page_content=entry, metadata={"source": "txt"}))
    print(f"✅ Loaded {len(documents)} Q/A documents from {txt_path}")
    return documents

def chunk_documents(documents, chunk_size=600, chunk_overlap=80):
    """
    Splits documents into overlapping chunks for embedding and retrieval.
    Returns: List[Document]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ".", " "]
    )
    all_chunks = []
    for doc in tqdm(documents, desc="Splitting Q/A"):
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

def build_chroma_index(
    chunks,
    persist_directory: str = "./chroma_store",
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cuda"
):
    """
    Embed and store chunks in ChromaDB using HuggingFaceEmbeddings and selected device.
    """
    os.makedirs(persist_directory, exist_ok=True)
    print(f"🧠 Loading embedding model: {embedding_model_name} on {device}")
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device}
    )
    print("⚙️ Building Chroma vector index...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist() # Writes RAM index to disk
    print(f"✅ Chroma store saved in: {persist_directory}")
    return vectordb

def load_chroma_vectorstore(persist_directory="./chroma_store", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    """
    Load an existing Chroma vector store for downstream RAG/agent use.
    """
    print(f"🧠 Loading embedding model: {embedding_model_name} on {device}")
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device}
    )
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    print(f"📦 Loaded Chroma vectorstore from {persist_directory}")
    return vectordb

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Chroma index from MedQuAD txt corpus w/ CUDA support")
    parser.add_argument("--txt_path", type=str, default="./docs/medquad_qa_corpus.txt", help="Path to MedQuAD txt corpus")
    parser.add_argument("--persist_dir", type=str, default="./chroma_store", help="Where to save Chroma index")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="HF embedding model name")
    parser.add_argument("--chunk_size", type=int, default=600)
    parser.add_argument("--chunk_overlap", type=int, default=80)
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU instead of CUDA")
    args = parser.parse_args()

    device = get_device(force_cpu=args.force_cpu)
    print(f"\n🌐 Embedding device: {device}\n")

    # Step 1: Load MedQuAD TXT
    documents = load_medquad_txt(args.txt_path)

    # Step 2: Chunk
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # Step 3: Build Index with Embedding, on chosen device
    build_chroma_index(
        chunks,
        persist_directory=args.persist_dir,
        embedding_model_name=args.embedding_model,
        device=device
    )

if __name__ == "__main__":
    main()
