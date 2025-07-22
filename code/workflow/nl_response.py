from .llm import get_llm, get_embedding_model
import re
from langchain_core.prompts import ChatPromptTemplate
from .report import report_generation_node

llm = get_llm()

def nl_response_node(state: dict) -> dict:
    """
    Generates a concise natural language response based on SQL results and user query.
    Falls back to report generation message if token limit is exceeded.
    """
    print("---GENERATING NATURAL LANGUAGE RESPONSE---")
      # or any token length approximation method

    user_query = state.get("user_query", "")
    sql_results = state.get("sql_results", [])

    # Combine all SQL result strings
    combined_result = "\n\n".join([r.get("result", "") for r in sql_results])

    # Token check (rough estimation)
    token_limit = 1500  # adjust based on model/token budget
    approx_token_count = len(combined_result.split())  # crude token estimate

    if approx_token_count > token_limit:
        print("⚠️ Result too large for summary, switching to report generation.")
        state["report"] = True
        state = report_generation_node(state)
        state["nl_response"] = "The result was too large to summarize. A downloadable report has been generated instead."
        return state

    # Step 1: Domain-specific response context
    domains = {m["domain_name"].lower().replace(" ", "_") for m in state.get("resolved_metrics", []) if m.get("domain_name")}
    response_instructions = ""
    for domain in domains:
        response_var = f"{domain}_response_system"
        response_text = globals().get(response_var, "").strip()
        if response_text:
            response_instructions += f"\n\n--- {domain.title()} Response Context ---\n{response_text}"

    # Step 2: Compose and format the prompt
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior care analytics assistant. Use the domain-specific context below to improve your responses.\n{response_instructions}"),
        ("human", "User Question:\n{user_query}\n\nSQL Results:\n{sql_result}\n\nAnswer the question based on the results.")
    ])

    prompt_value = response_prompt.format_prompt(
        response_instructions=response_instructions,
        user_query=user_query,
        sql_result=combined_result
    )

    response = llm.invoke(prompt_value)
    state["nl_response"] = response.content.strip()
    return state