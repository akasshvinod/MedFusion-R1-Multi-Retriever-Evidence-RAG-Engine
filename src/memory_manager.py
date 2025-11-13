"""
memory_manager.py
---------------------------------------------------------------
Manages conversational memory for DeepSeek-MCP-Medical-RAG system.

Features:
  - Short-term buffer memory (ChatMessageHistory)
  - Summarized long-term memory (LLM-powered)
  - Optional JSONL persistence (for reload/session)
  - Fully modular for RAG/chat/agent pipelines
"""

import os
import json
from datetime import datetime
from typing import Optional, Callable, List

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class MemoryManager:
    """
    Modular conversational memory system.
    Combines buffer, LLM summarization, and disk persistence.
    """

    def __init__(
        self,
        summary_trigger: int = 6,
        memory_file: str = "./memory_log.jsonl",
        summarizer_fn: Optional[Callable[[List], str]] = None
    ):
        """
        Args:
            summary_trigger: num of turns before summarizing
            memory_file: file for session persistence
            summarizer_fn: callable for LLM summarization (List[Message] -> str)
        """
        self.history = ChatMessageHistory()
        self.memory_file = memory_file
        self.summary_trigger = summary_trigger
        self.turn_count = 0
        self.summarizer_fn = summarizer_fn
        self.summary_buffer = []

        if os.path.exists(memory_file):
            print(f"📂 Reloading saved memory from {memory_file}")
            self.load_memory()

    # -----------------------------------------------------------
    def add_message(self, role: str, content: str):
        """Adds a message to memory and persists it. Summarizes if needed."""
        if role.lower() in ("user", "human"):
            self.history.add_user_message(content)
        elif role.lower() in ("ai", "assistant"):
            self.history.add_ai_message(content)
        elif role.lower() == "system":
            self.history.add_message(SystemMessage(content=content))
        else:
            raise ValueError(f"Unknown role: {role}")
        self.turn_count += 1

        self.save_to_disk(role, content)

        if self.summary_trigger and self.turn_count >= self.summary_trigger:
            self.summarize_memory()
            self.turn_count = 0

    # -----------------------------------------------------------
    def get_context(self, summarized: bool = False, n: Optional[int] = None) -> str:
        """
        Returns formatted context for RAG/LLM prompts, optionally last n turns and summaries.
        """
        msgs = self.history.messages if n is None else self.history.messages[-n:]
        context = "\n".join(
            [f"User: {m.content}" if isinstance(m, HumanMessage)
             else f"AI: {m.content}" if isinstance(m, AIMessage)
             else f"System: {m.content}" for m in msgs]
        )
        if summarized and self.summary_buffer:
            context = "\n".join(self.summary_buffer) + "\n" + context
        return context.strip()

    # -----------------------------------------------------------
    def summarize_memory(self):
        """
        Summarizes conversation using provided LLM/callback.
        Relies on summarizer_fn (List[Message] -> str).
        """
        if not self.summarizer_fn:
            print("⚠️ No summarizer function provided; skipping summarization.")
            return

        print("🧠 Summarizing conversation history...")
        prompt_messages = [
            SystemMessage(content="You are a clinical conversation summarizer."),
            HumanMessage(content="Summarize this conversation for continued medical safety and context:\n\n" + self.get_context())
        ]

        try:
            summary = self.summarizer_fn(prompt_messages)
            if hasattr(summary, 'content'):
                summary = summary.content
            elif isinstance(summary, list):
                summary = summary[0].content if summary else ""
            self.history.clear()
            self.history.add_ai_message(f"(Summary of previous chat): {summary}")
            self.summary_buffer.append(summary)
            print("✅ Memory summarized successfully.")
        except Exception as e:
            print(f"⚠️ Memory summarization failed: {e}")

    # -----------------------------------------------------------
    def save_to_disk(self, role: str, content: str):
        """
        Appends message to disk as JSONL for session persistence.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
        }
        try:
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Could not save memory to disk: {e}")

    def load_memory(self):
        """
        Loads chat history from disk and restores it in memory buffer.
        """
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    r, c = data["role"], data["content"]
                    if r.lower() in ("user", "human"):
                        self.history.add_user_message(c)
                    elif r.lower() in ("ai", "assistant"):
                        self.history.add_ai_message(c)
                    elif r.lower() == "system":
                        self.history.add_message(SystemMessage(content=c))
            print("✅ Memory restored from previous session.")
        except Exception as e:
            print(f"⚠️ Failed to load memory: {e}")

    def clear(self):
        """Clears history and deletes disk file."""
        self.history.clear()
        self.summary_buffer = []
        if os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
            except Exception as e:
                print(f"⚠️ Could not clear memory file: {e}")
        print("🧹 Memory cleared.")

# -----------------------------------------------------------
# CLI/Test Example
# -----------------------------------------------------------
if __name__ == "__main__":
    # ---- REAL LLM SUMMARIZER PLUGGED IN BELOW ----
    from deepseek_api import get_llm

    llm = get_llm()

    def summarizer_fn(messages):
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    mem = MemoryManager(summary_trigger=3, summarizer_fn=summarizer_fn)
    mem.add_message("user", "What are the symptoms of liver cancer?")
    mem.add_message("ai", "Liver cancer symptoms include fatigue and abdominal pain.")
    mem.add_message("user", "What are the treatment options?")
    mem.add_message("ai", "Treatment includes surgery, ablation, or immunotherapy.")
    print("\n🧠 Context Preview:\n", mem.get_context(summarized=True))
