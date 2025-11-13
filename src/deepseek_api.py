import os
from typing import List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

def get_llm(
    model: str = "tngtech/deepseek-r1t2-chimera:free",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    streaming: bool = False
) -> ChatOpenAI:
    """
    Initializes a LangChain ChatOpenAI LLM for OpenRouter models (DeepSeek/Chimera).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(" OPENROUTER_API_KEY not found in environment or .env!")
    return ChatOpenAI(
        model=model,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )
    
def run_llm(
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
    model: str = "tngtech/deepseek-r1t2-chimera:free",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    streaming: bool = False
) -> str:
    """
    Sends a structured message list to DeepSeek/Chimera model using OpenRouter/LangChain.
    Supports System/Human/AI message roles for prompt, context, and multi-turn chat.
    """
    llm = get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming
    )
    response = llm.invoke(messages)
    # Streaming/non-streaming output handling
    if hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    else:
        return str(response)
    
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
    # Demonstrate full chat context (system, user, AI, more user)
    full_dialogue = [
        SystemMessage(content="You are a knowledgeable, friendly medical assistant."),
        HumanMessage(content="What are the symptoms of hepato cellular carcinoma?"),
        AIMessage(content="Common symptoms include jaundice, fatigue, unexplained weight loss, and abdominal pain."),
        HumanMessage(content="How is it diagnosed?")
    ]
    try:
        output = run_llm(full_dialogue)
        print("🧠 DeepSeek Multi-Turn Response:\n", output)
    except Exception as e:
        print("❌ Error:", e)

