import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import ast
from pathlib import Path
# Load files
# Base directory where app.py lives
BASE_DIR = Path(__file__).resolve().parent

# File paths
customers_path = BASE_DIR / "Customers_2.csv"
transactions_path = BASE_DIR / "Transactions_2.csv"
ratings_path = BASE_DIR / "ProductVariationRatings_2.csv"
rules_path = BASE_DIR / "rules.json"

# Load the files
customers_df = pd.read_csv(customers_path)
transactions_df = pd.read_csv(transactions_path)
ratings_df = pd.read_csv(ratings_path)
rules_df = pd.read_json(rules_path, lines=True)

# Convert rule columns back to sets
rules_df['antecedents'] = rules_df['antecedents'].apply(set)
rules_df['consequents'] = rules_df['consequents'].apply(set)

# Function to find similar customers using cosine similarity
def find_similar_customers(input_customer, customer_base, top_n):
    numeric_cols = ['Age', 'Income', 'Credit Score', 'Tenure']
    categorical_col = ['Segment']

    X = customer_base[numeric_cols].copy()
    X_cat = customer_base[categorical_col]

    # Normalize numeric
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # One-hot encode segment
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X_cat)

    X_full = np.hstack([X_scaled, X_encoded])

    # Prepare input row
    input_df = pd.DataFrame([input_customer])
    input_scaled = scaler.transform(input_df[numeric_cols])
    input_encoded = encoder.transform(input_df[categorical_col])
    input_vector = np.hstack([input_scaled, input_encoded])

    # Compute cosine similarity
    similarities = cosine_similarity(input_vector, X_full)[0]
    similar_indices = np.argsort(similarities)[-top_n:][::-1]
    return customer_base.iloc[similar_indices].copy()

# Function to recommend next products using Apriori rules
def recommend_next_products(customer_products, rules_df, min_conf=0.5, min_lift=1.0):
    recommendations = set()
    for _, row in rules_df.iterrows():
        antecedent = list(row['antecedents'])
        consequent = list(row['consequents'])
        if set(antecedent).issubset(set(customer_products)) and row['confidence'] >= min_conf and row['lift'] >= min_lift:
            recommendations.update(consequent)
    return list(recommendations)

# Function to recommend best variation based on similar customers
def recommend_variation(product, similar_customers, ratings_df):
    candidates = ratings_df[(ratings_df['CustomerID'].isin(similar_customers['CustomerID'])) &
                            (ratings_df['Product'] == product)]
    if candidates.empty:
        return "Default"
    return candidates.groupby('Variation')['Rating'].mean().sort_values(ascending=False).idxmax()

# -------------------------
# Streamlit UI Starts Here
# -------------------------

st.title("🧠 Intelligent Product + Variation Recommender")

# Input customer details
st.header("Enter Customer Details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Income ($)", min_value=50000, max_value=500000, step=1000, value=120000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
with col2:
    tenure = st.number_input("Tenure (Years)", min_value=1, max_value=40, value=5)
    segment = st.selectbox("Segment", ["Mass Market", "Affluent", "Retiree"])
    main_product = st.selectbox("Current Product Owned", sorted(transactions_df['Product'].unique()))

# Recommendation Parameters
st.header("Recommendation Parameters")
min_conf = st.slider("Minimum Confidence", min_value=0.0, max_value=1.0, step=0.05, value=0.5, 
help="🔍 Confidence tells us how often the recommended product is bought when the original product is already purchased.\n\n"
        "E.g., if 75% of customers who have a Mortgage also have a Credit Card, then the rule:\n"
        "'Mortgage → Credit Card' has a confidence of 0.75.\n\n"
        "Default (0.5) means: recommend products only if at least 50% of similar customers also purchased them.")
min_lift = st.slider("Minimum Lift", min_value=0.5, max_value=5.0, step=0.1, value=1.0,
help= "📈 Lift measures how much more likely two products are bought together "
        "compared to random chance.\n\n"
        "E.g., a lift of 2.0 means customers are 2x more likely to buy both products together "
        "than if they were unrelated.\n\n"
        "Default: 1.0 (recommend only if there's at least neutral or positive association).")
top_k = st.slider("Number of Similar Customers", min_value=1, max_value=50, value=10)

# Initialize session_state on first run
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'similar_customers' not in st.session_state:
    st.session_state.similar_customers = None

# Submit
if st.button("🔍 Recommend"):
    # Step 1: Find similar customers
    input_customer = {
        "Age": age,
        "Income": income,
        "Credit Score": credit_score,
        "Tenure": tenure,
        "Segment": segment
    }
    similar_customers = find_similar_customers(input_customer, customers_df, top_k)

    # Step 2: Get recommended products (from Apriori)
    customer_products = [main_product]
    next_products = recommend_next_products(customer_products, rules_df, min_conf, min_lift)

    # Step 3: Recommend variation per product
    recommendations = []
    for prod in next_products:
        best_var = recommend_variation(prod, similar_customers, ratings_df)
        recommendations.append({
            "Recommended Product": prod,
            "Best Variation": best_var
        })

    
    # Save to session
    st.session_state.recommendations = recommendations
    st.session_state.similar_customers = similar_customers

    
# --- Display (always show if session_state has values) ---
if st.session_state.recommendations is not None:
    if st.session_state.recommendations:
        st.subheader("✅ Recommendations")
        st.table(pd.DataFrame(st.session_state.recommendations))
    else:
        st.warning("No product recommendations met the threshold.")

if st.session_state.similar_customers is not None:
    st.subheader(f"👥 Top {top_k} Similar Customers")
    st.caption("These are customers with similar age, income, credit score, tenure, and segment. Their ratings help personalize variation recommendations.")
    st.dataframe(st.session_state.similar_customers.reset_index(drop=True))

# --- Optional Clear Button ---
if st.button("❌ Clear"):
    st.session_state.recommendations = None
    st.session_state.similar_customers = None