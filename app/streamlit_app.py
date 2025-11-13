"""
streamlit_app.py
----------------
Enhanced Professional Streamlit Interface for Medical RAG System

Improvements:
- Better error handling and user feedback
- Performance optimizations
- Enhanced UI/UX
- Source display with metadata
- Query analytics
- Better session management
- Retry logic for failed requests
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import time
from typing import Dict, List, Optional

# Path resolution
script_path = Path(__file__).resolve()
script_dir = script_path.parent

if script_dir.name == "app":
    project_root = script_dir.parent
else:
    project_root = script_dir

src_path = project_root / "src"

if not src_path.exists():
    st.error(f"""
    ❌ **Cannot find src/ directory**
    
    **Expected location:** `{src_path}`
    **To fix:** Ensure src/ folder exists in `{project_root}`
    """)
    st.stop()

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from rag_agent_mcp import (
        get_medical_retriever,
        build_wikipedia_tool,
        build_pubmed_tool,
        MedicalRAGChain,
        intelligent_source_selection,
        heuristic_source_selection
    )
    from deepseek_api import get_llm
    from memory_manager import MemoryManager
    
except ImportError as e:
    st.error(f"""
    ❌ **Import Error**: {str(e)}
    
    Please ensure all required Python files exist in the src/ directory:
    - rag_agent_mcp.py
    - deepseek_api.py
    - memory_manager.py
    """)
    st.stop()

# Page config
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with dark mode support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        animation: fadeIn 0.3s ease-in;
        background-color: transparent;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background-color: rgba(30, 58, 138, 0.15);
        border-left-color: #3b82f6;
    }
    .assistant-message {
        background-color: rgba(34, 197, 94, 0.15);
        border-left-color: #22c55e;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .source-badge:hover {
        transform: scale(1.05);
    }
    .chroma-badge { background-color: #ffeb3b; color: #000; }
    .wikipedia-badge { background-color: #2196f3; color: #fff; }
    .pubmed-badge { background-color: #4caf50; color: #fff; }
    .stButton>button {
        width: 100%;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .source-detail {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .timestamp {
        color: #888;
        font-size: 0.75rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables with better defaults."""
    defaults = {
        'initialized': False,
        'messages': [],
        'rag_chain': None,
        'memory': None,
        'source_selections': [],
        'total_queries': 0,
        'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'query_times': [],
        'error_count': 0,
        'last_error': None,
        'retrieval_stats': {'chroma': 0, 'wikipedia': 0, 'pubmed': 0}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Cached initialization functions
@st.cache_resource(show_spinner=False)
def load_retriever(persist_dir: str, embedding_model: str, device: str, k: int):
    """Cache the retriever to avoid reloading."""
    try:
        return get_medical_retriever(
            persist_dir=persist_dir,
            embedding_model_name=embedding_model,
            device=device,
            k=k
        )
    except Exception as e:
        st.error(f"Failed to load retriever: {str(e)}")
        raise

@st.cache_resource(show_spinner=False)
def load_tools():
    """Cache Wikipedia and PubMed tools."""
    try:
        return build_wikipedia_tool(), build_pubmed_tool()
    except Exception as e:
        st.error(f"Failed to load external tools: {str(e)}")
        raise

def initialize_rag_system(persist_dir: str, embedding_model: str, device: str, 
                         k_retrieval: int, model_name: str, temperature: float, 
                         max_tokens: int, selection_mode: str, session_id: str):
    """Initialize the RAG system with comprehensive error handling."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load retriever
        status_text.text("Loading vector database...")
        progress_bar.progress(25)
        retriever_chroma = load_retriever(persist_dir, embedding_model, device, k_retrieval)
        
        # Step 2: Load tools
        status_text.text("Initializing external tools...")
        progress_bar.progress(50)
        wikipedia_tool, pubmed_tool = load_tools()
        
        # Step 3: Initialize LLM
        status_text.text("Connecting to language model...")
        progress_bar.progress(75)
        llm = get_llm(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )
        
        # Step 4: Setup memory and RAG chain
        status_text.text("Setting up conversation memory...")
        
        def summarizer_fn(messages):
            try:
                response = llm.invoke(messages)
                return response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                return f"Summary unavailable: {e}"
        
        memory = MemoryManager(
            summary_trigger=6,
            memory_file=f"./memory_{session_id}.jsonl",
            summarizer_fn=summarizer_fn
        )
        
        rag_chain = MedicalRAGChain(
            retriever_chroma=retriever_chroma,
            wikipedia_tool=wikipedia_tool,
            pubmed_tool=pubmed_tool,
            llm=llm,
            memory_manager=memory,
            use_intelligent_selection=(selection_mode == "Intelligent (LLM)")
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Initialization complete!")
        time.sleep(0.5)
        
        return rag_chain, memory
        
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        raise
    finally:
        progress_bar.empty()
        status_text.empty()

def format_sources_display(sources: Dict[str, bool], source_metadata: Optional[Dict] = None) -> str:
    """Create formatted HTML for source badges with metadata."""
    badges = []
    
    if sources.get("chroma"):
        tooltip = ""
        if source_metadata and "chroma_docs" in source_metadata:
            tooltip = f"Retrieved {len(source_metadata['chroma_docs'])} documents"
        badges.append(f'<span class="source-badge chroma-badge" title="{tooltip}">📚 Chroma</span>')
    
    if sources.get("wikipedia"):
        tooltip = ""
        if source_metadata and "wikipedia_result" in source_metadata:
            tooltip = "Medical encyclopedia content"
        badges.append(f'<span class="source-badge wikipedia-badge" title="{tooltip}">📖 Wikipedia</span>')
    
    if sources.get("pubmed"):
        tooltip = ""
        if source_metadata and "pubmed_result" in source_metadata:
            tooltip = "Research articles and papers"
        badges.append(f'<span class="source-badge pubmed-badge" title="{tooltip}">🔬 PubMed</span>')
    
    if badges:
        return f"<div style='margin-top: 0.5rem;'>Sources: {''.join(badges)}</div>"
    return ""

def export_chat_history() -> str:
    """Export chat history with metadata."""
    export_data = {
        "session_id": st.session_state.session_id,
        "export_timestamp": datetime.now().isoformat(),
        "total_queries": st.session_state.total_queries,
        "error_count": st.session_state.error_count,
        "retrieval_stats": st.session_state.retrieval_stats,
        "average_response_time": sum(st.session_state.query_times) / len(st.session_state.query_times) if st.session_state.query_times else 0,
        "messages": st.session_state.messages,
        "source_selections": st.session_state.source_selections
    }
    return json.dumps(export_data, indent=2)

# Initialize
init_session_state()

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🏥 Medical RAG System")
    st.markdown("---")
    
    # System Configuration
    with st.expander("⚙️ System Configuration", expanded=not st.session_state.initialized):
        persist_dir = st.text_input(
            "Chroma Directory",
            value="./chroma_store",
            help="Path to ChromaDB vector store"
        )
        
        # Auto-detect CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        
        device = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=0 if cuda_available else 1,
            help=f"GPU available: {'✅ Yes' if cuda_available else '❌ No'}"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-small-en-v1.5"
            ],
            help="HuggingFace embedding model"
        )
        
        k_retrieval = st.slider(
            "Documents to Retrieve (k)",
            min_value=1,
            max_value=10,
            value=3,
            help="More documents = better context but slower"
        )
    
    # Model Parameters
    with st.expander("🤖 Model Parameters", expanded=False):
        model_name = st.text_input(
            "Model",
            value="tngtech/deepseek-r1t2-chimera:free",
            help="OpenRouter model identifier"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.3,
            step=0.1,
            help="0 = Focused, 1 = Balanced, 2 = Creative"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=512,
            max_value=4096,
            value=2048,
            step=256,
            help="Maximum response length (higher = longer responses)"
        )
    
    # Source Selection Mode
    st.markdown("---")
    st.markdown("### 🔍 Retrieval Settings")
    
    selection_mode = st.radio(
        "Source Selection Mode",
        ["Intelligent (LLM)", "Heuristic", "Manual"],
        help="""
        - Intelligent: AI decides which sources to use
        - Heuristic: Rule-based source selection
        - Manual: You choose sources manually
        """
    )
    
    if selection_mode == "Manual":
        st.markdown("**Select Sources:**")
        use_chroma = st.checkbox("📚 Chroma (MedQuAD)", value=True)
        use_wikipedia = st.checkbox("📖 Wikipedia", value=True)
        use_pubmed = st.checkbox("🔬 PubMed", value=False)
    
    # Initialize System
    st.markdown("---")
    
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary"):
            try:
                rag_chain, memory = initialize_rag_system(
                    persist_dir=persist_dir,
                    embedding_model=embedding_model,
                    device=device,
                    k_retrieval=k_retrieval,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    selection_mode=selection_mode,
                    session_id=st.session_state.session_id
                )
                
                st.session_state.rag_chain = rag_chain
                st.session_state.memory = memory
                st.session_state.initialized = True
                
                st.success("✅ System initialized!")
                st.balloons()
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.session_state.error_count += 1
                st.session_state.last_error = str(e)
                st.error(f"❌ Initialization failed!")
                with st.expander("Error Details"):
                    st.exception(e)
    else:
        st.success("✅ System Ready")
        if st.button("🔄 Reinitialize", help="Restart the system with new settings"):
            st.session_state.initialized = False
            st.session_state.rag_chain = None
            st.session_state.memory = None
            st.rerun()
    
    # Session Statistics
    st.markdown("---")
    st.markdown("### 📊 Session Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Errors", st.session_state.error_count)
        avg_time = sum(st.session_state.query_times) / len(st.session_state.query_times) if st.session_state.query_times else 0
        st.metric("Avg Time", f"{avg_time:.1f}s")
    
    # Source usage stats
    if st.session_state.total_queries > 0:
        with st.expander("📈 Source Usage"):
            for source, count in st.session_state.retrieval_stats.items():
                pct = (count / st.session_state.total_queries) * 100
                st.write(f"**{source.title()}**: {count} ({pct:.0f}%)")
    
    # Session Management
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clear Chat", help="Clear all messages"):
            st.session_state.messages = []
            st.session_state.source_selections = []
            st.session_state.query_times = []
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.rerun()
    
    with col2:
        if st.session_state.messages:
            st.download_button(
                label="💾 Export",
                data=export_chat_history(),
                file_name=f"chat_{st.session_state.session_id}.json",
                mime="application/json",
                help="Download chat history as JSON"
            )

# Main content area
st.markdown('<h1 class="main-header">🏥 Medical RAG Assistant</h1>', unsafe_allow_html=True)

# Display warning if not initialized
if not st.session_state.initialized:
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ System Not Initialized</h3>
        <p>Configure settings in the sidebar and click <strong>"Initialize System"</strong> to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("### 🌟 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Multi-Source Retrieval**
        - 📚 MedQuAD Database (10K+ Q&A pairs)
        - 📖 Wikipedia Medical Content
        - 🔬 PubMed Research Papers
        - 🤖 Intelligent source selection
        """)
    
    with col2:
        st.markdown("""
        **🧠 Advanced AI Features**
        - 💭 Conversational memory
        - 🎯 Context-aware responses
        - 📝 Automatic summarization
        - 🔄 Streaming responses
        """)
    
    with col3:
        st.markdown("""
        **⚡ Performance & Quality**
        - 🚀 GPU acceleration support
        - ⚙️ Configurable parameters
        - 📊 Usage analytics
        - 💾 Export chat history
        """)
    
    st.markdown("---")
    
    # Quick start guide
    with st.expander("📖 Quick Start Guide", expanded=True):
        st.markdown("""
        **Getting Started:**
        
        1. **Configure System** (in sidebar):
           - Set ChromaDB path (default: `./chroma_store`)
           - Choose device (CUDA for GPU, CPU otherwise)
           - Select embedding model
        
        2. **Adjust Parameters** (optional):
           - Model: Choose your language model
           - Temperature: Control creativity (0.3 recommended for medical)
           - Max tokens: Set response length
        
        3. **Choose Retrieval Mode**:
           - **Intelligent**: AI automatically selects best sources
           - **Heuristic**: Rule-based selection
           - **Manual**: You control which sources to use
        
        4. **Initialize & Chat**:
           - Click "Initialize System"
           - Wait for confirmation
           - Start asking medical questions!
        
        **Example Questions:**
        - "What are the symptoms of diabetes?"
        - "Explain how antibiotics work"
        - "What is the latest research on COVID-19 treatments?"
        """)

else:
    # Medical Disclaimer
    with st.expander("⚕️ Important Medical Disclaimer"):
        st.warning("""
        **⚠️ Critical Information:**
        
        - 🚫 **NOT MEDICAL ADVICE**: This tool provides information only
        - 👨‍⚕️ **Consult Professionals**: Always seek qualified healthcare providers
        - 🚨 **Emergencies**: Call emergency services immediately (911/112)
        - 📅 **Currency**: Information may be outdated
        - ✅ **Verify**: Cross-check important medical information
        - 🔬 **Research**: Use for educational purposes only
        
        By using this tool, you acknowledge these limitations.
        """)
    
    # Chat interface
    st.markdown("### 💬 Conversation")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
            role = message["role"]
            content = message["content"]
            timestamp = message.get("timestamp", "")
            
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You</strong>
                    <span class="timestamp">{timestamp}</span><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Display sources
                sources_html = ""
                if idx < len(st.session_state.source_selections):
                    sources = st.session_state.source_selections[idx]
                    sources_html = format_sources_display(sources)
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant</strong>
                    <span class="timestamp">{timestamp}</span><br>
                    {content}
                    {sources_html}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("💬 Ask a medical question...", key="user_input")
    
    if user_input:
        start_time = time.time()
        current_time = datetime.now().strftime("%I:%M %p")
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": current_time
        })
        st.session_state.total_queries += 1
        st.session_state.memory.add_message("user", user_input)
        
        # Determine source selection
        try:
            if selection_mode == "Manual":
                selection = {
                    "chroma": use_chroma,
                    "wikipedia": use_wikipedia,
                    "pubmed": use_pubmed
                }
            elif selection_mode == "Heuristic":
                selection = heuristic_source_selection(user_input)
            else:  # Intelligent
                with st.spinner("🤔 Analyzing query..."):
                    selection = intelligent_source_selection(
                        user_input,
                        st.session_state.rag_chain.llm,
                        use_llm=True
                    )
            
            # Update stats
            for source, selected in selection.items():
                if selected:
                    st.session_state.retrieval_stats[source] += 1
            
            st.session_state.source_selections.append(selection)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Show selected sources
                source_names = [name.title() for name, selected in selection.items() if selected]
                
                with st.status(f"🔍 Searching: {', '.join(source_names)}...", expanded=False) as status:
                    full_response = ""
                    
                    try:
                        # Stream response
                        for chunk in st.session_state.rag_chain.invoke(user_input, stream=True):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        status.update(label="✅ Response complete", state="complete")
                        
                        # Add to messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "timestamp": datetime.now().strftime("%I:%M %p")
                        })
                        st.session_state.memory.add_message("ai", full_response)
                        
                        # Track time
                        query_time = time.time() - start_time
                        st.session_state.query_times.append(query_time)
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().strftime("%I:%M %p")
                        })
                        st.session_state.error_count += 1
                        st.session_state.last_error = str(e)
                        status.update(label="❌ Error occurred", state="error")
        
        except Exception as e:
            st.error(f"Query processing failed: {str(e)}")
            st.session_state.error_count += 1
        
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>
        <strong>Medical RAG System v2.0</strong><br>
        Powered by DeepSeek, LangChain & ChromaDB<br>
        🎓 For educational and research purposes only<br>
        Session ID: {session_id}
    </small>
</div>
""".format(session_id=st.session_state.session_id), unsafe_allow_html=True)