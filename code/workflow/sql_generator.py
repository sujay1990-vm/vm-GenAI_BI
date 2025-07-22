from .llm import get_llm, get_embedding_model
import re
from langchain_core.prompts import ChatPromptTemplate

llm = get_llm()

def sql_generation_node(state: dict, config: dict) -> dict:
    """
    Generates a unified Spark SQL query using resolved metrics and the global schema_metadata string.
    Handles retries using previous SQL attempts and error history.
    """
    print("🔥 sql_generation_node ENTERED")

    user_query = state["user_query"]
    resolved_metrics = state.get("resolved_metrics", [])
    error_history = state.get("error_history", [])
    metadata_store = state.get("metadata", {})
    # metadata_str = flatten_metadata_for_prompt(metadata_store)
    # ✅ Get domain names from metadata directly
    domains = metadata_store.keys()
    flattened_metadata = "\n\n".join([
    metadata_store[domain]["flattened_prompt"]
    for domain in domains if domain in metadata_store
        ])
    query_check_flag = state.get("query_check_flag", False)
    query_check_msg = state.get("query_check_msg", "")


    print(f"📌 Error history: {state.get('error_history')}")


    # Retry limit check
    MAX_SQL_ATTEMPTS = 5
    if len(error_history) >= MAX_SQL_ATTEMPTS:
        print(f"🛑 Max retry limit reached ({MAX_SQL_ATTEMPTS}). Exiting...")

        failure_prompt = f"""
        The system failed to generate a valid SQL query after {MAX_SQL_ATTEMPTS} attempts.

        User query: "{user_query}"

        Previous Attempts and Errors:
        {error_history[-1].split("Exception:")[-1].strip().split(';')[0]}

        As an assistant:
        1. Tell the user that the system could not generate a working SQL query.
        2. Explain the error in simple language, what might be causing it.
        Keep the message clear. Do NOT include SQL or attempt to answer the original question.
        """

        clarification_msg = llm.invoke(failure_prompt).content.strip()
        state["final_response"] = clarification_msg
        state["flow_exit_flag"] = True
        state["exit_reason"] = f"SQL generation failed after {MAX_SQL_ATTEMPTS} retries"
        return state

    # ---- 1. Format resolved metric blocks ----
    print("---GENERATING SQL QUERY---")
    metric_blocks = []
    for m in resolved_metrics:
        block = f"""- Metric: {m['metric']}
        • Definition: {m['definition']}
        • Formula: {m['formula']}"""
        metric_blocks.append(block)

    formatted_metrics = "\n\n".join(metric_blocks)

    # ---- 2. Visual/report flags ----
    visualize = state.get("visualize", False)
    report = state.get("report", False)
    vis_instructions = state.get("visual_instructions", "")

    # ---- 3. Retry context ----
    prior_errors = "\n".join(error_history) if error_history else "None"
    

    # ---- 4. SQL generation prompt ----
    sql_gen_system_prompt = f"""
    You are a Spark SQL generation assistant for senior care analytics.

    Your primary task is to generate a single, valid sqlite SQL query that accurately answers the user's question by transforming business metric definitions and formulas into correct SQL logic.
    You are doing this for Senior Assisted living long term care and their residents.
    Inputs:
    - User Query: {user_query}

    - Resolved Metrics:
    {formatted_metrics}

    - Full Schema Metadata (tables, columns, relationships, instructions, sample SQL queries):
    {flattened_metadata}

    - Previous Execution Errors that contain user query, previous sql and error faced:
    {prior_errors}

    <IMPORTANT NOTE>
        Generate sqlite queries only
        Generate a syntactically correct SQL query in plain text (no triple backticks).
        Return ONLY the SQL without any commentary.
    </IMPORTANT NOTE>
    Instructions:
    - Strict SQL rules:
        - Never apply aggregate functions (e.g., SUM, AVG) to primary key columns or unique identifiers like ResidentKey, EmployeeID, or RecordID.
        - Only count such fields using COUNT(DISTINCT ...) when counting entities.
        - Do not use SUM/AVG on categorical fields or non-numeric IDs.
        - If the year is not specified in the user query, ALWAYS default to the current calendar year (e.g., WHERE YEAR = YEAR(CURRENT_DATE)).
        - Always distinguish between FacilityName and LocationName:
        - FacilityName examples: "Parker at Somerset", "Parker at River Road"
        - LocationName examples: "Somerset", "River Road", "Stonegate"
        - Always default to using LocationName when filtering data by place.
        - Only use FacilityName if LocationName is not available in the relevant table.


    - Carefully convert the **plain English formulas and definitions** into valid SQL expressions.
    - Join tables properly based on provided entity relationships.
    - Always use the correct aggregation and filtering logic described in the metric.
    - If `visualize` is true, group the result appropriately (e.g., by time, physician, or facility).
    - If `report` is true, format the output for tabular export (alias columns, add sorting if needed).
    - If there are previous SQL attempts or execution errors, ensure you do not repeat the same mistakes.
    ⚠️ DO NOT join two fact tables directly unless a 1-to-1 relationship is guaranteed.
        If you need data from two fact tables:
        - Aggregate each fact table separately using a CTE or subquery.
        - Then JOIN the aggregated results using appropriate dimension keys.
        For example, do NOT do this:
        JOIN Fact_Census fc ON Fact_Budget.LocationKey = fc.LocationKey
    {f"- WARNING: The last SQL attempt was rejected due to validation errors: {query_check_msg}" if query_check_flag else ""}

    {f"Visualization instruction: {vis_instructions}" if visualize else ""}
    Return ONLY the final SQL query — no commentary, no markdown.
    """.strip()


    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", sql_gen_system_prompt)
    ])

    # ---- 5. LLM call ----
    prompt_value = sql_prompt.format_prompt()
    sql_response = llm.invoke(prompt_value).content.strip()
    # Clean common LLM formatting issues
    sql_response = re.sub(r"^```sql\s*", "", sql_response, flags=re.IGNORECASE)
    sql_response = re.sub(r"^```", "", sql_response, flags=re.IGNORECASE)
    sql_response = re.sub(r"```$", "", sql_response, flags=re.IGNORECASE)
    sql_response = re.sub(r"^sql\b[\s\n]*", "", sql_response, flags=re.IGNORECASE)

    sql_response = sql_response.strip()
    # ---- 6. Postprocess SQL (split if needed) ----
    stmts = re.split(r';\s*(?=SELECT)', sql_response, flags=re.IGNORECASE)
    stmts = [s.rstrip(';').strip() + ';' for s in stmts if s.strip()]

    # Cache it
    state["sql_queries"] = stmts
    # state["sql_cache_key"] = cache_key  # ✅ add this for executor to use

    return state


