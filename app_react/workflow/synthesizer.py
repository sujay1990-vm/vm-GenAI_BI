from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from llm import get_llm

llm = get_llm()


@tool
def synthesizer_tool(user_query: str, sql_outputs: List[str] = [], rag_outputs: List[str] = []) -> str:
    """
    Synthesizes a final answer based on SQL results and RAG (document) context.
    Returns a clear, accurate natural language response.
    """
    print("🧠 Synthesizing final output...")

    rag_context = "\n\n".join(rag_outputs) if rag_outputs else "None"
    sql_context = "\n\n".join(sql_outputs) if sql_outputs else "None"

    prompt = [
        SystemMessage(content="""You are an assistant for insurance knowledge management. Use the provided SQL results and/or document context—which may include claims data, policy documents, and regulatory guidelines—to answer the user's question accurately and clearly. Do not hallucinate or include information that is not in the context."""),

        HumanMessage(content=f"""SQL Results:
{sql_context}

RAG Context:
{rag_context}

Question: {user_query}

Answer in a clear, helpful, and factual way using the context above.
""")
    ]

    response = llm.invoke(prompt)
    return response.content