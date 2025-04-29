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
conn = sqlite3.connect("cross_selling.db", check_same_thread=False)
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
def fetch_schema_info(dummy_input: str = "") -> str:
    """
    Returns the pre‐built SCHEMA_INFO string describing your database schema.
    """
    return SCHEMA_INFO

@tool
def run_sql(sql_statements: str) -> str:
    """
    Executes one or more semicolon-separated SQL statements against `conn`.
    - SELECT queries: returns top 5 rows as a markdown table.
    - Other queries: returns success/failure status.
    """
    outputs = []
    for stmt in sql_statements.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            if stmt.lower().startswith("select"):
                df = pd.read_sql(stmt, conn)
                if df.empty:
                    outputs.append(f"✅ `{stmt}` returned no rows.")
                else:
                    table = df.to_markdown(index=False)
                    outputs.append(f"✅ Results for `{stmt}`:\n\n{table}")
            else:
                conn.execute(stmt)
                conn.commit()
                outputs.append(f"✅ Executed `{stmt}` successfully.")
        except Exception as e:
            outputs.append(f"❌ Error executing `{stmt}`:\n{str(e)}")
    return "\n\n".join(outputs)


@tool
def clean_sql_statement(raw_sql: str) -> str:
    """
    Strips out markdown fences (```sql``` or ```), any leading/trailing backticks,
    and excessive whitespace so you get a clean SQL string.
    """
    # Remove opening fences like ```sql or ``` and closing ```
    cleaned = re.sub(r"```(?:sql)?\s*", "", raw_sql)
    cleaned = cleaned.replace("```", "")
    # Collapse any multiple newlines/spaces
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    # Remove trailing semicolon if desired (we’ll handle semis in run_sql)
    cleaned = cleaned.strip().rstrip(";")
    return cleaned

import json, pandas as pd

# --- one-time loads ---------------------------------------------
seg_map = pd.read_csv("segment_map.csv").set_index("Customer_ID")["segment"]
takeup  = pd.read_parquet("segment_takeup.parquet")

@tool
def segment_rate_score(input: str) -> str:
    """
    JSON  {"customer_id": "...", "product_ids": [...]}  →
    JSON  {"P003": 0.071, "P005": 0.018, ...}
    Uses k-means clusters built from rich behavioural features.
    """
    req  = json.loads(input)
    cid  = req["customer_id"]
    pids = req["product_ids"]

    if cid not in seg_map:
        return json.dumps({p:0.0 for p in pids})

    seg = int(seg_map[cid])
    if seg not in takeup.index:
        return json.dumps({p:0.0 for p in pids})

    rates = takeup.loc[seg]
    return json.dumps({p: round(float(rates.get(p,0.0)),3) for p in pids})


@tool
def filter_eligible_products(input: str) -> str:
    """
    Input  ► JSON {"customer_id": "...", "product_ids": ["P003","P005",...]}
    Output ► JSON {"eligible": [...], "ineligible": {product: reason, ...}}
    Deterministic screen using `eligibility_rules.csv` + already-owned check.
    """
    import json, pandas as pd
    payload = json.loads(input)
    cid     = payload["customer_id"]
    cands   = payload["product_ids"]

    cust    = pd.read_sql(f"SELECT Age, Annual_Income, Credit_Score "
                          f"FROM customers WHERE Customer_ID='{cid}'", conn).iloc[0]
    owned   = pd.read_sql(f"SELECT Product_ID FROM customer_products "
                          f"WHERE Customer_ID='{cid}'", conn)["Product_ID"].tolist()

    rules   = eligibility_df.set_index("Product_ID").loc[cands]
    eligible, ineligible = [], {}
    for pid, row in rules.iterrows():
        fails = []
        if pd.notna(row.min_age)          and cust.Age          < row.min_age:          fails.append("age")
        if pd.notna(row.min_income)       and cust.Annual_Income< row.min_income:       fails.append("income")
        if pd.notna(row.min_credit_score) and cust.Credit_Score < row.min_credit_score: fails.append("credit")

        excl = row.get("excludes", "")
        if excl and any(o in excl.split(",") for o in owned):
            fails.append("exclusion_conflict")

        if pid in owned:
            fails.append("already_owned")

        if fails:
            ineligible[pid] = ",".join(fails)
        else:
            eligible.append(pid)
    return json.dumps({"eligible": eligible, "ineligible": ineligible})
#################################################

import json, pandas as pd, sqlite3
from langchain.agents import tool
####################################
# DB connection reused by all tools
conn = sqlite3.connect("cross_selling.db", check_same_thread=False)

# ---------- association rules JSON ---------------
with open("synergy_rules.json", "r") as f:
    raw_rules = json.load(f)

# Convert lists → sets for fast lookup
rules = [
    {
        "antecedents": set(r["antecedents"]),
        "consequents": set(r["consequents"]),
        "lift": float(r["lift"])
    }
    for r in raw_rules
]

# ---------- cache customer->owned products -------
customer_owned = (
    pd.read_sql("SELECT Customer_ID, Product_ID FROM customer_products", conn)
      .groupby("Customer_ID")["Product_ID"].agg(list)
      .to_dict()
)


@tool
def lift_score(input: str) -> str:
    """
    Input  ►  JSON {"customer_id": "...",
                    "product_ids": ["P003","P005",...]}
    Output ►  JSON {"P003": 1.82, "P005": 1.00, ... }

    • Uses pre-computed association rules (Apriori) to measure how strongly
      each candidate pairs with ANY product the customer already owns.
    • Score = maximum lift; returns 1.00 if no rule exists.
    """
    pl    = json.loads(input)
    cid   = pl["customer_id"]
    cands = pl["product_ids"]

    owned = customer_owned.get(cid, [])
    scores = {}

    for pid in cands:
        lifts = [
            r["lift"]
            for r in rules
            if (pid in r["consequents"]
                and any(o in r["antecedents"] for o in owned))
        ]
        scores[pid] = round(max(lifts) if lifts else 1.00, 3)

    return json.dumps(scores)
 ###########################################

import json, joblib, numpy as np
from langchain.agents import tool

# ── one-time loads ───────────────────────────────────────────
als       = joblib.load("als_model.joblib")
row_idx   = joblib.load("user_index.joblib")   # Customer_ID → row #
col_idx   = joblib.load("item_index.joblib")   # Product_ID  → col #
# Build reverse map for safety
item_rev  = {v: k for k, v in col_idx.items()}

@tool
def als_score(input: str) -> str:
    """
    Input  ► JSON {"customer_id": "...",
                   "product_ids": ["P003","P014",...]}
    Output ► JSON {"P003": 2.43, "P014": 0.91, ...}
    Score = latent dot-product (higher = stronger preference)
    """
    req  = json.loads(input)
    cid  = req["customer_id"]
    pids = req["product_ids"]

    # If unseen customer, return zeros
    if cid not in row_idx:
        return json.dumps({p: 0.0 for p in pids})

    uvec = als.user_factors[row_idx[cid]]

    scores = {}
    for p in pids:
        col = col_idx.get(p)
        if col is None:
            scores[p] = 0.0         # never seen in training
            continue
        ivec = als.item_factors[col]
        scores[p] = round(float(np.dot(uvec, ivec)), 3)

    # Optional min-max rescale so range ≈ 0–1 within this set
    if scores:
        mn, mx = min(scores.values()), max(scores.values())
        if mx > mn:                                     # avoid /0
            for p in scores:
                scores[p] = round((scores[p]-mn)/(mx-mn), 3)

    return json.dumps(scores)

import json, pandas as pd
from langchain.agents import tool

# ---------- already-loaded helpers ----------
#  • filter_eligible_products
#  • segment_rate_score
#  • lift_score
#  • als_score
#  • transition_lift_score
#  (all must be in your tools list)

@tool
def recommend_bundle(input: str) -> str:
    """
    Input  ► JSON {"customer_id": "...", "top_n": 3}
    Output ► JSON list with objects:
             [{"product_id":"P003",
               "eligibility":"yes",
               "segment_rate":0.07,
               "lift":1.82,
               "als":0.63,
               "seq_lift":1.25,
               "composite":0.08}, ...]  (top N)
    Blend rule ► composite = segment_rate * lift * als * seq_lift
    """
    # ---- parse -----
    req    = json.loads(input)
    cid    = req["customer_id"]
    cands    = req["product_ids"]
    top_n  = req.get("top_n", 3)
    # allids = pd.read_csv("products.csv")["Product_ID"].tolist()

    # # 1) eligibility
    # elig   = json.loads(filter_eligible_products(
    #           json.dumps({"customer_id":cid,"candidate_ids":pids})))
    # cands  = elig["eligible"]

    # if not cands:
    #     return json.dumps([])

    # 2) scores
    seg  = json.loads(segment_rate_score(json.dumps(
              {"customer_id":cid,"product_ids":cands})))
    lift = json.loads(lift_score(json.dumps(
              {"customer_id":cid,"product_ids":cands})))
    als  = json.loads(als_score(json.dumps(
              {"customer_id":cid,"product_ids":cands})))


    # 3) composite
    rows = []
    for pid in cands:
        comp = seg[pid] * lift[pid] * als[pid]
        rows.append({
            "product_id": pid,
            "segment_rate": seg[pid],
            "lift": lift[pid],
            "als": als[pid],
            "composite": round(comp,5)
        })

    ranked = sorted(rows, key=lambda x: x["composite"], reverse=True)[:top_n]
    return json.dumps(ranked)



fetch_customer_profile.description = "Fetch the customer's demographic and financial profile by full name."

fetch_product_catalog.description = "Retrieve the complete product catalog including features, target behaviors, eligibility criteria, and special offers."

fetch_owned_products.description = "Get a list of products already owned by the customer to avoid duplicate recommendations."

scientific_calculator.description = "Perform numeric calculations such as averages, ratios, or thresholds to support financial reasoning."

fetch_schema_info.description = (
    "Use this tool to retrieve the database schema (tables, columns, types) "
    "Use this tool for any doubts with tables"
)

run_sql.description = (
    "Use this tool to execute raw SQL statements. "
    "Accepts one or multiple semicolon-separated queries. "
    "Use this tool to resolve data related doubts in Thoughts"
    "use fetch_schema tool before using this tool"
)

analyze_customer_behavior.description = (
    "► Purpose\n"
    "    Produce a concise, machine-readable snapshot of a customer’s "
    "behavioural KPIs for recommendation reasoning.\n\n"
    "► When to call\n"
    "    • You ALREADY have the Customer_ID.\n"
    "    • You need overall spend totals, top category, idle balance, etc.\n"
    "    • NOT for raw transaction queries or SQL look-ups.\n\n"
    "► Input\n"
    "    The single Customer_ID as a plain string (e.g. \"CUST0008\").\n\n"
    "► Output\n"
    "    JSON summary, e.g. "
    "{\"Total_Spend\": 9453.12, \"Top_Spend_Category\": \"Grocery\", "
    "\"Idle_Balance_Est\": 1100.0, \"Aggregation_Days\": 60 }.\n"
)


clean_sql_statement.description = (
    "Use this tool to take any raw SQL (even wrapped in markdown fences) "
    "and return a single clean SQL string, ready for execution." )

filter_eligible_products.description = (
    "► Purpose\n"
    "    Hard-screen a list of candidate product IDs, removing anything the\n"
    "    customer is not legally or sensibly allowed to open.\n\n"
    "► When to call\n"
    "    • Right after you generate (or fetch) a pool of potential offers.\n"
    "    • ALWAYS run this BEFORE any scoring or ranking tools.\n\n"
    "► Input (JSON string)\n"
    "    {\"customer_id\": \"CUST0008\",\n"
    "     \"product_ids\": [\"P003\", \"P005\", ...] }\n"
    "► Output (JSON string)\n"
    "    {\"eligible\":   [\"P003\", \"P014\", ...],\n"
    "     \"ineligible\": {\"P005\": \"income,credit\", ...}}\n\n"
    "► Rule set\n"
    "    Reads eligibility_rules.csv (min_age, min_income, min_credit_score,\n"
    "    excludes list) **and** the customer’s current holdings to decide\n"
    "    eligibility deterministically.\n"
    "    Already-owned or rule-violating products are listed in “ineligible”."
)


segment_rate_score.description = (
    "► Purpose\n"
    "    Look up the HISTORICAL take-up probability for one or more candidate "
    "products, using the customer’s k-means behavioural cluster (8 clusters).\n\n"
    "► When to call\n"
    "    • You already know the customer_id AND a list of candidate product_ids.\n"
    "    • You need a quick probability proxy BEFORE ranking or bundling.\n\n"
    "► Input (JSON string)\n"
    "    {\"customer_id\": \"CUST0008\", "
    "     \"product_ids\": [\"P003\", \"P005\", \"P014\"] }\n\n"
    "► Output (JSON string)\n"
    "    {\"P003\": 0.071, \"P005\": 0.018, \"P014\": 0.042 }\n\n"
    "► Notes for the LLM\n"
    "    • Pass EXACTLY those two keys: customer_id, product_ids.\n"
    "    • If a product_id is missing from the segment table, the tool returns 0.0.\n"
)

lift_score.description = (
    "► Purpose\n"
    "    Estimate how well each CANDIDATE product pairs with the customer's "
    "    EXISTING portfolio, using historical association-rule lift.\n\n"
    "► When to call\n"
    "    • You already know customer_id and a list of ELIGIBLE products.\n"
    "    • You want to rank or boost products that have strong co-purchase "
    "      history with what the customer already owns.\n\n"
    "► Input  (JSON string)\n"
    "    {\"customer_id\": \"CUST0012\", "
    "     \"product_ids\": [\"P003\",\"P006\",\"P014\"] }\n\n"
    "► Output (JSON string)\n"
    "    {\"P003\": 1.82, \"P006\": 1.00, \"P014\": 2.07}\n\n"
    "► Notes\n"
    "    • A score >1 means the product historically appears MORE often with "
    "      the customer's owned items than by chance.\n"
    "    • A score =1 means no observed lift (neutral)."
)


als_score.description = (
    "Returns a collaborative-filtering preference score for each candidate "
    "product, computed via an ALS latent-factor model. Input JSON must contain "
    "customer_id and product_ids. Scores are rescaled 0–1 within the candidate "
    "list; higher means the product matches the customer's latent interests."
)

recommend_bundle.description = (
    "Primary recommendation engine. Give it "
    "{\"customer_id\":\"CUST0010\",\"top_n\":3,"
    "     \"product_ids\": [\"P003\",\"P006\",\"P014\"] }\n\n"
    "It auto-calls eligibility, segment score, lift score, ALS score and "
    "sequential lift, multiplies them, and returns the top N products with "
    "all component scores for explanation."
)


from dotenv import load_dotenv
import os, openai

load_dotenv()          # reads .env into os.environ
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


OPENAI_DEPLOYMENT_ENDPOINT = "https://advancedanalyticsopenaikey.openai.azure.com/" 
OPENAI_API_VERSION = "2024-12-01-preview"

OPENAI_DEPLOYMENT_NAME = "gpt-4o"
OPENAI_MODEL_NAME="gpt-4o"


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
from langchain.agents import AgentExecutor, create_react_agent
import textwrap

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", react_prompt),
        ("user", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ]
)

# tools = [fetch_customer_profile, analyze_customer_behavior, fetch_product_catalog, fetch_owned_products, fetch_schema_info, run_sql ,clean_sql_statement ]
tools = [fetch_schema_info, run_sql  , clean_sql_statement,analyze_customer_behavior,filter_eligible_products,segment_rate_score, lift_score, als_score]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt   # Optional. If omitted, uses LangChain's default ReAct prompt
)

# 4. Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)



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
