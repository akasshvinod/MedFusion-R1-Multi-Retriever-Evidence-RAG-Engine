"""
rag_agent_mcp.py
----------------
Production-Ready Medical RAGChain with Memory, Multi-retriever, Error Handling, and Smart Selection.

Features:
- Parallel retrieval from Chroma (MedQuAD), Wikipedia (API), PubMed abstracts
- LLM-driven source selection for intelligent retrieval
- Conversational memory integrated into prompts
- Robust error handling with fallbacks
- Token-aware context truncation
- Streaming with response capture
- Weighted source prioritization

Usage:
    python src/rag_agent_mcp.py --persist_dir ../chroma_store --device cuda --stream
"""

import logging
import json
from typing import Dict, List, Optional, Callable
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from prompts import get_prompt

try:
    from src.deepseek_api import get_llm
except ImportError:
    from deepseek_api import get_llm

try:
    from src.rag_build_index_chroma import load_chroma_vectorstore
except ImportError:
    from rag_build_index_chroma import load_chroma_vectorstore

try:
    from src.memory_manager import MemoryManager
except ImportError:
    from memory_manager import MemoryManager

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pymed import PubMed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_agent_mcp")


class SafeRetriever:
    """Wrapper for safe retrieval with error handling and fallbacks."""
    
    def __init__(self, retriever_fn: Callable, source_name: str, fallback: str = ""):
        self.retriever_fn = retriever_fn
        self.source_name = source_name
        self.fallback = fallback
    
    def __call__(self, *args, **kwargs) -> str:
        try:
            result = self.retriever_fn(*args, **kwargs)
            if not result:
                logger.warning(f"{self.source_name}: No results found")
                return self.fallback
            return result
        except Exception as e:
            logger.error(f"{self.source_name} retrieval failed: {e}")
            return self.fallback


def pubmed_search(query: str, email: str = "akasshvinod@gmail.com", max_results: int = 3) -> str:
    """Search PubMed for medical literature."""
    try:
        pubmed = PubMed(tool="LangChainMedicalAgent", email=email)
        results = pubmed.query(query, max_results=max_results)
        output = []
        
        for article in results:
            title = article.title or "Untitled"
            pmid = article.pubmed_id or "N/A"
            abstract = article.abstract or "No abstract available."
            
            # Truncate abstract sensibly
            if len(abstract) > 400:
                abstract = abstract[:400].rsplit(' ', 1)[0] + "..."
            
            output.append(f"**{title}** (PMID: {pmid})\n{abstract}\n")
        
        return "\n".join(output) if output else ""
    except Exception as e:
        logger.error(f"PubMed search error: {e}")
        return ""


def get_medical_retriever(
    persist_dir: str = "./chroma_store",
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cpu",
    k: int = 3,
):
    """Initialize Chroma vectorstore retriever."""
    try:
        embedding = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device}
        )
        vectordb = load_chroma_vectorstore(
            persist_directory=persist_dir,
            embedding_model_name=embedding_model_name,
            device=device
        )
        return vectordb.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logger.error(f"Failed to load Chroma retriever: {e}")
        raise


def build_wikipedia_tool():
    """Build Wikipedia retrieval tool."""
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    
    def wikipedia_search(question: str) -> str:
        try:
            result = wiki.run(question)
            # Truncate at sentence boundary
            if len(result) > 1200:
                result = result[:1200].rsplit('.', 1)[0] + '.'
            return result
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return ""
    
    return wikipedia_search


def build_pubmed_tool(pubmed_email: str = "akasshvinod@gmail.com"):
    """Build PubMed retrieval tool."""
    return lambda question: pubmed_search(question, email=pubmed_email)


def intelligent_source_selection(question: str, llm, use_llm: bool = True) -> Dict[str, bool]:
    """
    Use LLM to intelligently determine which sources to query.
    Falls back to heuristic-based selection if LLM fails.
    """
    if not use_llm:
        return heuristic_source_selection(question)
    
    try:
        selector_prompt = """Analyze this medical question and determine which sources to query.

Question: {question}

Sources available:
- chroma: Local medical Q&A database (MedQuAD)
- wikipedia: General medical encyclopedia
- pubmed: Recent medical research papers

Respond ONLY with valid JSON (no markdown, no explanation):
{{"chroma": true/false, "wikipedia": true/false, "pubmed": true/false}}

Guidelines:
- Use chroma for general medical questions, symptoms, conditions
- Use wikipedia for definitions, overviews, basic concepts
- Use pubmed for latest research, clinical trials, specific studies
- You can select multiple sources

JSON response:"""
        
        response = llm.invoke(selector_prompt.format(question=question))
        content = response.content if hasattr(response, "content") else str(response)
        
        # Clean potential markdown formatting
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        selection = json.loads(content)
        logger.info(f"LLM source selection: {selection}")
        return selection
        
    except Exception as e:
        logger.warning(f"LLM source selection failed: {e}. Using heuristic fallback.")
        return heuristic_source_selection(question)


def heuristic_source_selection(question: str) -> Dict[str, bool]:
    """Fallback heuristic-based source selection."""
    question_lower = question.lower()
    
    # Keywords for different sources
    research_keywords = ["latest", "recent", "research", "study", "trial", "published"]
    general_keywords = ["what is", "define", "explain", "overview", "introduction"]
    clinical_keywords = ["symptom", "treatment", "diagnosis", "cause", "prevent"]
    
    selection = {
        "chroma": any(kw in question_lower for kw in clinical_keywords) or len(question.split()) < 10,
        "wikipedia": any(kw in question_lower for kw in general_keywords),
        "pubmed": any(kw in question_lower for kw in research_keywords)
    }
    
    # If nothing selected, default to chroma + wikipedia
    if not any(selection.values()):
        selection = {"chroma": True, "wikipedia": True, "pubmed": False}
    
    logger.info(f"Heuristic source selection: {selection}")
    return selection


def format_chat_history(memory_manager: MemoryManager, n: int = 8) -> str:
    """Format recent chat history for context using MemoryManager's get_context method."""
    try:
        # Use the built-in get_context method with last n messages
        context = memory_manager.get_context(summarized=True, n=n)
        return context if context else "No previous conversation."
    except Exception as e:
        logger.warning(f"Error formatting chat history: {e}")
        return "No previous conversation."


def chroma_retriever_to_text(docs: List[Document]) -> str:
    """Convert Chroma documents to text with metadata."""
    if not docs:
        return ""
    
    output = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        source = doc.metadata.get("source", "Unknown")
        output.append(f"[Source {i}: {source}]\n{content}")
    
    return "\n\n".join(output)


class MedicalRAGChain:
    """Production-ready Medical RAG Chain with memory and multi-source retrieval."""
    
    def __init__(
        self,
        retriever_chroma,
        wikipedia_tool,
        pubmed_tool,
        llm,
        memory_manager: MemoryManager,
        use_intelligent_selection: bool = True
    ):
        self.retriever_chroma = retriever_chroma
        self.wikipedia_tool = SafeRetriever(wikipedia_tool, "Wikipedia", "")
        self.pubmed_tool = SafeRetriever(pubmed_tool, "PubMed", "")
        self.llm = llm
        self.memory = memory_manager
        self.use_intelligent_selection = use_intelligent_selection
        
        # Source weights for context prioritization
        self.source_weights = {
            "chroma": 0.45,
            "pubmed": 0.35,
            "wikipedia": 0.20
        }
    
    def _build_chain(self, selection: Dict[str, bool]):
        """Build the RAG chain dynamically based on source selection."""
        # Get the retrieval prompt
        prompt_template = get_prompt("retrieval")
        
        # Check if chat_history is in the template
        use_chat_history = "{chat_history}" in prompt_template
        
        if use_chat_history:
            # Modify prompt to include chat history if not already present
            if "Previous conversation:" not in prompt_template:
                prompt_template = prompt_template.replace(
                    "{question}",
                    "Previous conversation:\n{chat_history}\n\nCurrent question: {question}"
                )
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "chat_history"]
            )
        else:
            # Simpler prompt without chat history
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        
        # Build parallel retrieval branches
        parallel_branches = {"question": RunnablePassthrough()}
        
        if selection.get("chroma", False):
            parallel_branches["chroma"] = (
                self.retriever_chroma 
                | RunnableLambda(chroma_retriever_to_text)
            )
        
        if selection.get("wikipedia", False):
            parallel_branches["wikipedia"] = RunnableLambda(self.wikipedia_tool)
        
        if selection.get("pubmed", False):
            parallel_branches["pubmed"] = RunnableLambda(self.pubmed_tool)
        
        # Build the chain with or without chat history
        if use_chat_history:
            chain = (
                RunnableParallel(parallel_branches)
                | RunnableLambda(lambda d: {
                    "context": self._format_context(d),
                    "question": d["question"],
                    "chat_history": format_chat_history(self.memory)
                })
                | prompt
                | self.llm
                | RunnableLambda(lambda x: x.content if hasattr(x, "content") else str(x))
            )
        else:
            chain = (
                RunnableParallel(parallel_branches)
                | RunnableLambda(lambda d: {
                    "context": self._format_context(d),
                    "question": d["question"]
                })
                | prompt
                | self.llm
                | RunnableLambda(lambda x: x.content if hasattr(x, "content") else str(x))
            )
        
        return chain
    
    def _format_context(self, retrieval_results: Dict[str, str]) -> str:
        """Format context from multiple sources with weights and structure."""
        sections = []
        
        # Order by weight (highest first)
        ordered_sources = sorted(
            self.source_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for source, weight in ordered_sources:
            content = retrieval_results.get(source, "").strip()
            if content:
                source_label = source.upper()
                sections.append(f"=== {source_label} (Relevance: {weight:.0%}) ===\n{content}")
        
        if not sections:
            return "No relevant information found from retrieval sources."
        
        return "\n\n".join(sections)
    
    def invoke(self, question: str, stream: bool = False):
        """Process a question through the RAG pipeline."""
        # Determine which sources to use
        selection = intelligent_source_selection(
            question,
            self.llm,
            use_llm=self.use_intelligent_selection
        )
        
        logger.info(f"Selected sources: {[k for k, v in selection.items() if v]}")
        
        # Build and execute chain
        chain = self._build_chain(selection)
        
        if stream:
            return chain.stream(question)
        else:
            return chain.invoke(question)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Production Medical RAGChain with Memory and Intelligent Retrieval"
    )
    parser.add_argument("--persist_dir", type=str, default="../chroma_store")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--k", type=int, default=3, help="Number of documents to retrieve from Chroma")
    parser.add_argument("--pubmed_email", type=str, default="akasshvinod@gmail.com")
    parser.add_argument("--model", type=str, default="tngtech/deepseek-r1t2-chimera:free", help="Model name for LLM")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for LLM response")
    parser.add_argument("--stream", action="store_true", help="Enable streaming responses")
    parser.add_argument("--simple_selection", action="store_true", help="Use heuristic instead of LLM for source selection")
    args = parser.parse_args()
    
    # Initialize components
    logger.info("Initializing RAG components...")
    
    try:
        retriever_chroma = get_medical_retriever(
            persist_dir=args.persist_dir,
            embedding_model_name=args.embedding_model,
            device=args.device,
            k=args.k,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Chroma retriever: {e}")
        return
    
    wikipedia_tool = build_wikipedia_tool()
    pubmed_tool = build_pubmed_tool(pubmed_email=args.pubmed_email)
    
    # Initialize LLM with proper parameters for your deepseek_api
    llm = get_llm(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        streaming=args.stream
    )
    
    # Create summarizer function for MemoryManager
    def summarizer_fn(messages):
        """Summarizer function for memory manager."""
        try:
            response = llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Summary unavailable."
    
    # Initialize memory with summarizer
    memory = MemoryManager(
        summary_trigger=6,
        memory_file="./memory_log.jsonl",
        summarizer_fn=summarizer_fn
    )
    
    # Create RAG chain
    rag_chain = MedicalRAGChain(
        retriever_chroma=retriever_chroma,
        wikipedia_tool=wikipedia_tool,
        pubmed_tool=pubmed_tool,
        llm=llm,
        memory_manager=memory,
        use_intelligent_selection=not args.simple_selection
    )
    
    logger.info("✅ RAG system initialized successfully")
    print("\n" + "="*60)
    print("Medical RAG System Ready")
    print("="*60)
    print("Commands: 'exit' to quit, 'clear' to reset memory")
    print("="*60 + "\n")
    
    # Interactive loop
    while True:
        try:
            user_query = input("\n🏥 User> ").strip()
            
            if not user_query:
                continue
            
            if user_query.lower() in ("exit", "quit"):
                print("\nGoodbye! 👋")
                break
            
            if user_query.lower() == "clear":
                memory.clear()
                print("✅ Memory cleared")
                continue
            
            # Add to memory
            memory.add_message("user", user_query)
            
            # Process query
            if args.stream:
                print("\n🤖 Assistant> ", end="", flush=True)
                chunks = []
                for chunk in rag_chain.invoke(user_query, stream=True):
                    print(chunk, end="", flush=True)
                    chunks.append(chunk)
                print()  # New line after streaming
                response = "".join(chunks)
            else:
                response = rag_chain.invoke(user_query, stream=False)
                print(f"\n🤖 Assistant> {response}")
            
            # Add response to memory
            memory.add_message("ai", response)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit properly.")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()