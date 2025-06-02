from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from graph import GraphState

def synthesizer(state: GraphState):
    print("🧠 Synthesizing final output...")

    user_query = state["user_query"]
    rag_chunks: list[Document] = state.get("rag_outputs", [])
    sql_outputs: list[str] = state.get("sql_outputs", [])

    rag_context = "\n\n".join([doc.page_content for doc in rag_chunks]) if rag_chunks else "None"
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
    return {"final_response": response.content}