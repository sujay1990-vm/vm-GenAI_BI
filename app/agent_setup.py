import sqlite3
import pandas as pd
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pathlib import Path
from prompts import *


from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
import textwrap
import re
# Base directory where app.py resides
BASE_DIR = Path(__file__).resolve().parent

# Correct path to the DB
DB_PATH = BASE_DIR / "cross_selling.db"

@tool
def fetch_customer_profile(name: str) -> str:
    """Fetch basic customer profile by full name."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM customers WHERE First_Name || ' ' || Last_Name = '{name}'", conn)
    return df.to_json(orient="records") if not df.empty else "Customer not found."

@tool
def analyze_customer_behavior(customer_id: str) -> str:
    """Provides a detailed analysis of customer behavior, spending patterns, and financial signals."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM feature_store WHERE Customer_ID = '{customer_id}'", conn)
        if df.empty:
            return "No behavior data found for this customer."
        
        row = df.iloc[0]
        insights = []

        # 1️⃣ Aggregation Period
        insights.append(f"Analysis based on {row['Aggregation_Days']} days of transaction data.")

        # 2️⃣ Spending Overview
        insights.append(f"Total spending during this period is ${row['Total_Spend']:.2f} across {row['Num_Transactions']} transactions.")
        insights.append(f"Average transaction amount is ${row['Avg_Txn_Amount']:.2f}, with a maximum single transaction of ${row['Max_Txn_Amount']:.2f}.")
        
        # 3️⃣ Key Spending Categories
        category_flags = []
        if row.get("Spend_Grocery", 0) > 500:
            category_flags.append(f"Grocery: ${row['Spend_Grocery']:.2f}")
        if row.get("Spend_Travel", 0) > 800:
            category_flags.append(f"Travel: ${row['Spend_Travel']:.2f}")
        if row.get("Spend_Fuel", 0) > 150:
            category_flags.append(f"Fuel: ${row['Spend_Fuel']:.2f}")
        if row.get("Spend_Medical", 0) > 200:
            category_flags.append(f"Medical: ${row['Spend_Medical']:.2f}")
        if row.get("Spend_Entertainment", 0) > 300:
            category_flags.append(f"Entertainment: ${row['Spend_Entertainment']:.2f}")

        if category_flags:
            insights.append("Significant spending detected in categories: " + "; ".join(category_flags))
        else:
            insights.append(f"Primary spending category is {row['Top_Spend_Category']}.")

        # 4️⃣ Income & Salary Patterns
        if row["Has_Salary_Credit"]:
            insights.append("Regular salary deposits detected, indicating stable income.")
        if row["Salary_to_Spend_Ratio"] > 0.5:
            insights.append(f"Healthy disposable income, with a Salary-to-Spend Ratio of {row['Salary_to_Spend_Ratio']:.2f}.")

        # 5️⃣ Financial Profile
        if row["Annual_Income"] > 100000:
            insights.append(f"High annual income: ${row['Annual_Income']}.")
        elif row["Annual_Income"] > 60000:
            insights.append(f"Moderate annual income: ${row['Annual_Income']}.")

        if row["Credit_Score"] >= 750:
            insights.append(f"Excellent credit score: {row['Credit_Score']}.")
        elif row["Credit_Score"] >= 700:
            insights.append(f"Good credit score: {row['Credit_Score']}.")
        else:
            insights.append(f"Credit score is {row['Credit_Score']}.")

        # 6️⃣ Spend Variability
        if row["Spend_Variability"] > 500:
            insights.append(f"High variability in spending, suggesting inconsistent transaction amounts.")
        else:
            insights.append(f"Consistent spending behavior with low variability.")

        # 7️⃣ Idle Balance Potential
        if row["Idle_Balance_Estimate"] > 5000:
            insights.append(f"Estimated idle balance of ${row['Idle_Balance_Estimate']:.2f}, indicating potential for savings or investment products.")

    return " | ".join(insights)

@tool
def fetch_product_catalog(dummy_input: str) -> str:
    """Returns the bank's product catalog for cross-selling."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM products", conn)
    return df.to_json(orient="records")

@tool
def fetch_owned_products(customer_id: str) -> str:
    """
    Fetches the list of products currently owned by the customer.
    The LLM should avoid recommending these products again.
    """
    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
            SELECT p.Product_ID, p.Product_Name, p.Product_Type
            FROM customer_products cp
            JOIN products p ON cp.Product_ID = p.Product_ID
            WHERE cp.Customer_ID = '{customer_id}'
        """
        df = pd.read_sql(query, conn)
    
    if df.empty:
        return "This customer does not own any products currently."
    
    # Return a readable list
    owned_list = df[['Product_ID', 'Product_Name', 'Product_Type']].to_dict(orient='records')
    return f"Customer currently owns the following products: {owned_list}"

@tool
def scientific_calculator(expression: str) -> str:
    """Performs safe scientific calculations. Provide expressions like '1250 / 28' or 'sqrt(256)'."""
    import math
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed_names['abs'] = abs

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# Define your schema, metadata, and relationships

@tool
def text_to_sql(user_query: str) -> str:
    """
    Generates and executes an SQL query based on the user's natural language question.
    Returns the query result or an appropriate message.
    """
    # Combine schema info with user query
    sql_prompt = f"""
    You are a SQL assistant. Based on the following schema, generate an accurate SQL query based on schema info.

    {SCHEMA_INFO}

    User Question: {user_query}

    Respond ONLY with the SQL query.
    """

    # Call LLM to generate SQL (assuming llm object exists)
    sql_query = llm.invoke(sql_prompt).content.strip()

    # Execute SQL safely
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df_result = pd.read_sql(sql_query, conn)
        if df_result.empty:
            return f"Query executed successfully, but no results found.\nSQL: {sql_query}"
        return f"SQL: {sql_query}\n\nResult:\n{df_result.head(5).to_string(index=False)}"
    except Exception as e:
        return f"Error executing SQL.\nGenerated Query: {sql_query}\nError: {str(e)}"

fetch_customer_profile.description = "Fetch the customer's demographic and financial profile by full name."

fetch_product_catalog.description = "Retrieve the complete product catalog including features, target behaviors, eligibility criteria, and special offers."

fetch_owned_products.description = "Get a list of products already owned by the customer to avoid duplicate recommendations."

scientific_calculator.description = "Perform numeric calculations such as averages, ratios, or thresholds to support financial reasoning."

text_to_sql.description = (
    "Use this tool for ANY question involving numbers, totals, spending amounts, transaction details, "
    "lists of products, or customer data insights. If the user asks 'how much', 'list', 'show', or refers "
    "to categories like fuel, groceries, travel, etc., ALWAYS use this tool."
)

analyze_customer_behavior.description = (
    "Use this ONLY to summarize customer behavior patterns for making product recommendations. "
    "Do NOT use this tool to answer specific data questions like amounts spent."
)


OPENAI_DEPLOYMENT_ENDPOINT = "https://az-openai-document-question-answer-service.openai.azure.com/" 
OPENAI_API_KEY = "5d24331966b648738e5003caad552df8" 
OPENAI_API_VERSION = "2023-05-15"

OPENAI_DEPLOYMENT_NAME = "az-gpt_35_model"
OPENAI_MODEL_NAME="gpt-3.5-turbo"

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = "az-embedding_model" 
OPENAI_ADA_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

encoding_name = "cl100k_base"

llm = AzureChatOpenAI(
                        temperature=0.1,
                        deployment_name=OPENAI_DEPLOYMENT_NAME,
                        model_name=OPENAI_MODEL_NAME,
                        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                        openai_api_version=OPENAI_API_VERSION,
                        openai_api_key=OPENAI_API_KEY            
                    )


from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
import textwrap

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [fetch_customer_profile, analyze_customer_behavior, fetch_product_catalog, scientific_calculator, fetch_owned_products ]
llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Pretty print with wrapping at 100 characters

def clean_llm_output(text):
    # Remove newline characters that break words (single letters per line)
    cleaned_lines = []
    lines = text.split('\n')

    buffer = ""
    for line in lines:
        if len(line.strip()) == 1:
            buffer += line.strip()
        else:
            if buffer:
                cleaned_lines.append(buffer)
                buffer = ""
            cleaned_lines.append(line)
    if buffer:
        cleaned_lines.append(buffer)

    cleaned_text = "\n".join(cleaned_lines)

    # Optionally fix any extra spaces
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)  # Ensure max double line breaks

    return cleaned_text


def fix_vertical_text(output):
    # Remove unintended newlines after Benefit, Reason, etc.
    fixed = re.sub(r'(Benefit:)\s*\n+', r'\1 ', output)
    fixed = re.sub(r'(Reason:)\s*\n+', r'\1 ', fixed)
    fixed = re.sub(r'(Eligibility Criteria:)\s*\n+', r'\1 ', fixed)
    return fixed
