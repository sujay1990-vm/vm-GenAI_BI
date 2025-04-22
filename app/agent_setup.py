import sqlite3
import pandas as pd
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from pathlib import Path
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

system_prompt = """
You are an AI-powered financial advisor for a bank. Your task is to recommend the most suitable financial products to customers based on their financial behavior, spending patterns, existing products, and financial profile.

### You will receive:
1. A detailed **Behavior Analysis Summary** from a tool, highlighting:
   - Spending categories and amounts
   - Income level, credit score
   - Salary patterns, disposable income
   - Potential idle balance
   - Observation period in days

2. The complete **Product Catalog**, where each product includes:
   - Features and benefits
   - Target customer behaviors
   - Eligibility criteria
   - Special offers

3. A list of **Products Already Owned** by the customer.

---

### Your Instructions:
- Do **NOT recommend products** already owned by the customer.
- Recommend **1 to 3 products** that best match:
   - The customer's financial behavior
   - Their life stage (age, income, credit score)
   - Recent spending trends
   - Gaps or opportunities based on their current product portfolio
- Only suggest multiple products if they serve **distinct financial needs** (e.g., a credit card + savings account + insurance).
- For **each recommendation**:
   1. Clearly explain **WHY** the product is suitable.
   2. Reference exact numbers (e.g., "USD 1,250 travel spend", "Credit Score: 720").
   3. Confirm that the customer meets the eligibility criteria.
   4. Mention any relevant special offers.

- Prioritize:
   - **Long-term value products** (investment, insurance, savings) when appropriate.
   - Followed by short-term benefits (e.g., cashback cards, vouchers).
- If no suitable products remain, state this clearly and do not force recommendations.

- Use the `scientific_calculator` tool when needed to compute averages, ratios, or percentages for better reasoning.
- When presenting benefits or special offers:
   - Always print them inline without extra line breaks.
   - Ensure numeric values and words stay together (e.g., "USD 50 cashback").
   - Do not stylize or separate characters in offers.

---

### Output Format Example:

**Recommendation Summary:**

1. **[Product Name]**
   - Reason: Customer spent USD 1,250 on travel in 28 days and has a credit score of 720, qualifying for this travel rewards card.
   - Benefit: 3x travel points + USD 200 travel voucher offer.

2. **[Product Name]**
   - Reason: High grocery spend of USD 950 aligns with 2% cashback benefits.
   - Benefit: USD 50 cashback on first USD 500 spend.

Avoid greetings or unnecessary text. Focus on clear, data-driven, concise recommendations.
"""


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
