import sqlite3
import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from llm import get_llm
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated, List
import operator
import re
from langchain_core.tools import tool

class SQLWorkerState(TypedDict):
    query: str
    sql_outputs: list[str]


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

@tool
def sql_worker_tool(query: str) -> str:
    """
    Executes a Spark/SQL query against the database.
    Accepts one or more SQL SELECT statements and returns the results.
    """
    print(f"🛠️ SQL Worker received query:\n{query}")

    # 1. Clean up SQL query (remove markdown formatting)
    sql_clean = re.sub(r"^```sql\s*|```$", "", query.strip(), flags=re.IGNORECASE)
    sql_clean = re.sub(r"^sql\b[\s\n]*", "", sql_clean, flags=re.IGNORECASE)

    # 2. Split into multiple statements if needed
    stmts = re.split(r';\s*(?=SELECT)', sql_clean, flags=re.IGNORECASE)
    stmts = [stmt.rstrip(';').strip() + ';' for stmt in stmts if stmt.strip()]

    # 3. Execute SQL
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
                    results.append(f"ℹ️ No results for:\n```sql\n{stmt}\n```")
                else:
                    formatted = "\n".join([str(dict(zip(col_names, row))) for row in rows])
                    results.append(f"✅ Results for:\n```sql\n{stmt}\n```\n{formatted}")

            except Exception as e:
                results.append(f"❌ SQL Error:\n```sql\n{stmt}\n```\n{str(e)}")

        cursor.close()
        conn.close()

    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

    return "\n\n".join(results)