import sqlite3
import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from llm import get_llm

llm = get_llm()
# Load schema and metric definitions once
with open("new_schema.json", "r") as f:
    table_schema = json.load(f)

with open("vocab_dictionary.json", "r") as f:
    metric_definitions = json.load(f)

# Prompt to guide SQL generation
sql_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL assistant for insurance claims and policy data.
Use only the provided schema and metric definitions to answer the user's query with correct SQL.
Do not hallucinate table or column names.

Schema:
{schema}

Metric Definitions:
{definitions}

User Question: {query}

Provide only the SQL query.""")
])

# Chain to convert NL to SQL
sql_chain = sql_gen_prompt | llm | StrOutputParser()

def sql_worker(state: SQLWorkerState) -> dict:
    query = state["query"]
    print(f"🛠️ SQL Worker received query: {query}")

    # SQL Generation
    schema_str = json.dumps(table_schema, indent=2)
    definition_str = json.dumps(metric_definitions, indent=2)
    sql_response  = sql_chain.invoke({
        "schema": schema_str,
        "definitions": definition_str,
        "query": query
    }).strip()

    # 2. Clean LLM artifacts
    sql_clean = re.sub(r"^```sql\s*", "", sql_response, flags=re.IGNORECASE)
    sql_clean = re.sub(r"^```", "", sql_clean, flags=re.IGNORECASE)
    sql_clean = re.sub(r"```$", "", sql_clean, flags=re.IGNORECASE)
    sql_clean = re.sub(r"^sql\b[\s\n]*", "", sql_clean, flags=re.IGNORECASE)
    sql_clean = sql_clean.strip()

    # 3. Split multiple statements if needed
    stmts = re.split(r';\s*(?=SELECT)', sql_clean, flags=re.IGNORECASE)
    stmts = [stmt.rstrip(';').strip() + ';' for stmt in stmts if stmt.strip()]

    # 4. Execute and collect results
    results = []
    try:
        conn = sqlite3.connect("my_database.db")
        cursor = conn.cursor()

        for stmt in stmts:
            try:
                cursor.execute(stmt)
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []

                if not rows:
                    results.append(f"No results for:\n```sql\n{stmt}\n```")
                else:
                    formatted_rows = "\n".join([str(dict(zip(col_names, row))) for row in rows])
                    results.append(f"Results for:\n```sql\n{stmt}\n```\n{formatted_rows}")

            except Exception as e:
                results.append(f"❌ SQL Execution Error:\n```sql\n{stmt}\n```\nError: {str(e)}")

        cursor.close()
        conn.close()

    except Exception as e:
        results = [f"❌ Connection Error: {str(e)}"]

    # 5. Return all SQL outputs
    return {"sql_outputs": results}