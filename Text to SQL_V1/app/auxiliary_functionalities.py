import json
import pandas as pd
import time
import pytz
import datetime

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import tiktoken
# from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores.faiss import FAISS
from api_key import *

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

from prompts.quest_and_ans_prompt_template import *
import fitz  # PyMuPDF
import os
from langchain.prompts import ChatPromptTemplate
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
import os
import shutil
from typing import List, Tuple
from transformers import GPT2Tokenizer
import re
import sqlite3
"""
Auxiliary functions
"""
# Returns the number of tokens in a text string.

"""
Get embedding model and LLM chat model
"""
def get_embedding():
    embedding_model = AzureOpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                                            deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                            model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                            azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT)
    return embedding_model

def get_llm_chat_model():
    llm_chat_model = AzureChatOpenAI(
                        temperature=0.1,
                        deployment_name=OPENAI_DEPLOYMENT_NAME,
                        model_name=OPENAI_MODEL_NAME,
                        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                        openai_api_version=OPENAI_API_VERSION,
                        openai_api_key=OPENAI_API_KEY               
                    )
    return llm_chat_model


#############################################################################


## Create FAISS Store for Loss Cause

def create_faiss_store(csv_file, embedding_model):
    """
    Create a FAISS vector store for `loss_cause` values from a CSV file.

    :param csv_file: Path to the CSV file containing the `loss_cause` column
    :param embedding_model: Embedding model instance for generating embeddings
    :return: FAISS vector store containing `loss_cause` values
    """
   
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Get unique non-null `loss_cause` values
    unique_loss_causes = df['loss_cause'].dropna().unique().tolist()

    # Generate embeddings for the unique loss causes
    loss_cause_embeddings = embedding_model.embed_documents(unique_loss_causes)

    # Create the FAISS vector store
    faiss_store = FAISS.from_texts(
        texts=unique_loss_causes,
        embedding=embedding_model
    )

    print("FAISS vector store created successfully.")
    return faiss_store


# Load CSV into SQLite
def load_csv_to_sqlite(csv_file, db_file, table_name):
    conn = sqlite3.connect(db_file)
    df = pd.read_csv(csv_file, encoding='latin-1')
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Data from {csv_file} loaded into SQLite table {table_name}.")


# Define a custom schema
def generate_table_schema():
    schema = [
        {"Column Name": "claim_no", "Data Type": "VARCHAR", "Null": "NO", "Primary Key": "YES"},
        {"Column Name": "fnol_call", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "LossDate", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "loss_cause", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "loss_location_zip", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "claim_description", "Data Type": "TEXT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "LOB", "Data Type": "VARCHAR", "Null": "NO", "Primary Key": "NO"},
        {"Column Name": "PolicyNumber", "Data Type": "VARCHAR", "Null": "NO", "Primary Key": "NO"},
        {"Column Name": "direct_paid_loss", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "direct_paid_LAE", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "direct_outstanding_loss", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "CAT_flag", "Data Type": "BOOLEAN", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "loss_year", "Data Type": "INTEGER", "Null": "NO", "Primary Key": "NO"},
        {"Column Name": "Age_max", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "CoverageA_Limit", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "ConstructionYear", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "DriverCount", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "CreditScore", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "bi_limit_1", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "bi_limit_2", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "one_auto_full_cov_flag", "Data Type": "BOOLEAN", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "Age_min", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "MaritalStatus", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "Gender", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "VehicleCount", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "AgencyProductNam", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "EarnedPremium", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "AnnualizedIPAmt", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "minPolicyEffectiveDt", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "minPolicyInceptionDts", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "maxPolicyExpirationDt", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "maxPolicyCancelDt", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "IncurredLossCat", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "IncurredLossNonCat", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "adjuster_notes", "Data Type": "TEXT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "num_adjuster_notes", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"}
    ]
    
    # Format schema as a string for LLM input
    formatted_schema = "\n".join(
        f"Column Name: {col['Column Name']}, Data Type: {col['Data Type']}, Null: {col['Null']}, Primary Key: {col['Primary Key']}"
        for col in schema
    )
    return formatted_schema

def create_table_metadata():
    metadata = [
        {"Column Name": "claim_no", "Description": "Unique identifier for the insurance claim."},
        {"Column Name": "fnol_call", "Description": "First Notice of Loss call identifier or details."},
        {"Column Name": "LossDate", "Description": "Date of the reported loss."},
        {"Column Name": "loss_cause", "Description": "Cause of the loss (e.g., fire, theft)."},
        {"Column Name": "loss_location_zip", "Description": "ZIP code where the loss occurred."},
        {"Column Name": "claim_description", "Description": "Detailed description of the claim."},
        {"Column Name": "LOB", "Description": "Line of Business associated with the claim."},
        {"Column Name": "PolicyNumber", "Description": "Policy number linked to the claim."},
        {"Column Name": "direct_paid_loss", "Description": "Amount directly paid for the loss."},
        {"Column Name": "direct_paid_LAE", "Description": "Amount directly paid for Loss Adjustment Expenses."},
        {"Column Name": "direct_outstanding_loss", "Description": "Outstanding amount for the loss."},
        {"Column Name": "CAT_flag", "Description": "Indicates if the claim is related to a catastrophe event."},
        {"Column Name": "loss_year", "Description": "Year when the loss occurred."},
        {"Column Name": "Age_max", "Description": "Maximum age of the insured person(s)."},
        {"Column Name": "CoverageA_Limit", "Description": "Limit of Coverage A for the policy."},
        {"Column Name": "ConstructionYear", "Description": "Year the insured property was constructed."},
        {"Column Name": "DriverCount", "Description": "Number of drivers covered under the policy."},
        {"Column Name": "CreditScore", "Description": "Credit score of the insured person or entity."},
        {"Column Name": "bi_limit_1", "Description": "First limit for bodily injury coverage."},
        {"Column Name": "bi_limit_2", "Description": "Second limit for bodily injury coverage."},
        {"Column Name": "one_auto_full_cov_flag", "Description": "Indicates if one auto has full coverage."},
        {"Column Name": "Age_min", "Description": "Minimum age of the insured person(s)."},
        {"Column Name": "MaritalStatus", "Description": "Marital status of the insured person."},
        {"Column Name": "Gender", "Description": "Gender of the insured person."},
        {"Column Name": "VehicleCount", "Description": "Number of vehicles covered under the policy."},
        {"Column Name": "AgencyProductNam", "Description": "Name of the agency product linked to the policy."},
        {"Column Name": "EarnedPremium", "Description": "Earned premium amount for the policy."},
        {"Column Name": "AnnualizedIPAmt", "Description": "Annualized installment premium amount."},
        {"Column Name": "minPolicyEffectiveDt", "Description": "Earliest effective date of the policy."},
        {"Column Name": "minPolicyInceptionDts", "Description": "Earliest inception date of the policy."},
        {"Column Name": "maxPolicyExpirationDt", "Description": "Latest expiration date of the policy."},
        {"Column Name": "maxPolicyCancelDt", "Description": "Latest cancellation date of the policy."},
        {"Column Name": "IncurredLossCat", "Description": "Incurred loss amount for catastrophe claims."},
        {"Column Name": "IncurredLossNonCat", "Description": "Incurred loss amount for non-catastrophe claims."},
        {"Column Name": "adjuster_notes", "Description": "Notes or comments added by the claim adjuster."},
        {"Column Name": "num_adjuster_notes", "Description": "Number of notes made by the adjuster."}
    ]
    
    # Format metadata as a string for LLM input
    formatted_metadata = "\n".join(
        f"Column Name: {col['Column Name']}, Description: {col['Description']}" for col in metadata
    )
    return formatted_metadata

class ConversationManager:
    def __init__(self, history_limit=5):
        """
        Initialize the conversation manager with a history limit.
        """
        self.history = []
        self.history_limit = history_limit

    def update_conversation_history(self, question, answer):
        """
        Store previous questions and answers, but limit the history size.
        """
        self.history.append(f"Question: {question} \n Answer: {answer}")
        if len(self.history) > self.history_limit:
            self.history.pop(0)  # Remove the oldest entry to keep the history within the limit

    def get_conversation_context(self):
        """
        Get the full conversation context as a string.
        """
        return self.history
    
    def clear_history(self):
        """
        Clear the conversation history to erase context.
        """
        self.history = []

    def reformulate_question(self, query_text, llm):
        """
        Reformulate the follow-up question based on the conversation history.
        """
               # Use the reformulation prompt to generate a standalone question
        reformulation_prompt = """
        Given a chat history ( User Questions and LLM response ) and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is.

        Chat History:
        {chat_history}

        Latest Question:
        {latest_question}
        """

        # Prepare the input for the LLM model
        formatted_prompt = reformulation_prompt.format(
            chat_history=self.get_conversation_context(),
            latest_question=query_text
        )

        # Get the reformulated question from LLM
        reformulated_response = llm.invoke(input=formatted_prompt)
        standalone_question = reformulated_response.content.strip()

        return standalone_question


def preprocess_sql_query(sql_query, faiss_store, embedding_model):
    """
    Standardize loss_cause in the SQL query using the FAISS vector store.
    
    :param sql_query: The SQL query generated by the LLM
    :param faiss_store: The FAISS vector store containing loss_cause embeddings
    :param embedding_model: The embedding model to create embeddings for user input
    :return: The updated SQL query with standardized loss_cause
    """
    import re

    # Check for loss_cause condition in the query
    match = re.search(r"loss_cause\s*(ILIKE|LIKE|=)\s*'([^']+)'", sql_query, re.IGNORECASE)
    if match:
        # Extract the condition (LIKE, ILIKE, =) and the value to be standardized
        condition = match.group(1).upper()
        loss_cause_value = match.group(2)

        # Query the FAISS vector store for the closest match
        query_embedding = embedding_model.embed_query(loss_cause_value)
        matches = faiss_store.similarity_search_by_vector(query_embedding, k=1)

        if matches:
            closest_match = matches[0].page_content  # Get the closest standardized value
            print(f"Standardized loss_cause: {closest_match}")

            # Handle LIKE or ILIKE conditions with wildcards
            if "LIKE" in condition:
                standardized_value = f"%{closest_match}%"
            else:
                standardized_value = closest_match

            # Replace the original loss_cause value in the query
            standardized_query = re.sub(
                r"loss_cause\s*(ILIKE|LIKE|=)\s*'([^']+)'",
                f"loss_cause {condition} '{standardized_value}'",
                sql_query,
                flags=re.IGNORECASE,
            )
            return standardized_query
        else:
            print("No close match found in FAISS for loss_cause.")
    else:
        print("No loss_cause condition found in the query.")

    # Return the original query if no changes were made
    return sql_query



# Execute the SQL query
def execute_sql_query_with_preprocessing(db_file, sql_query, faiss_store, embedding_model):
    # Preprocess the SQL query for loss_cause
    print(f"Original SQL query: {sql_query}")
    standardized_sql_query = preprocess_sql_query(sql_query, faiss_store, embedding_model)
    print(f"standard SQL query:{standardized_sql_query}")
    # Connect to the database and execute the query
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute(standardized_sql_query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return pd.DataFrame(result, columns=columns)
    except sqlite3.Error as e:
        conn.close()
        return f"SQL error: {e}"





# Generate SQL Query using AzureChatOpenAI LLM
def generate_sql_query(llm, natural_language_query, table_metadata, table_name, schema):
    query_prompt = f"""
You are a SQL expert. Use the table schema and metadata provided below to generate a SQL query. The data contains information about Insurance Claims, FNOL call, Adjustor notes and other details. 

<table info>

Table: {table_name}
Schema:
{schema}
Metadata: 
{table_metadata}

<table info/>

<Special Instructions>
- The SQL query must extract data semantically from text-based columns like 'adjuster_notes' , 'fnol_call'
without relying on exact pattern matching (e.g., LIKE or CONTAINS).
- Summarize or Explain means to read the entire Unstructured / Text Column requested
- claim no. , Claim Number , claim # - all mean the same primary key column
- Use LIMIT instead of TOP for restricting rows.

<Special Instructions/>

Convert the following natural language query into a SQL query:

{natural_language_query}

Only return the SQL query, no additional text.
"""
    response = llm.invoke(query_prompt)
    return response.content

# Execute the SQL query
def execute_sql_query(db_file, sql_query):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return pd.DataFrame(result, columns=columns)
    except sqlite3.Error as e:
        conn.close()
        return f"SQL error: {e}"
    
# Generate Natural Language Response using LLM
def generate_response(llm, dataframe, standalone_question, chat_history):
    """
    Generate a concise natural language response based on the standalone question,
    the SQL query result, and the chat history. Incorporate special instructions for uniform interpretation of key terms.
    """
    special_instructions = """
Interpret the following terms consistently:
- "Summarize," "Describe," and "Explain" all mean "Provide a detailed breakdown with comprehensive coverage of the topic."
- Focus on clarity, detail, and coherence in the response.
- Avoid generic phrases or filler text (e.g., "Based on the information provided").
- Do not assume missing or unrelated information unless explicitly stated in the data.
"""
    
    if isinstance(dataframe, str) and "SQL error" in dataframe:
    # Handle SQL syntax errors
        error_prompt = f"""
    The following request could not be processed due to a technical error in generating or executing the SQL query.
    Explain this to the user in simple terms without using technical jargon like 'SQL error' or 'query execution.'
    Focus on a helpful and user-friendly response.

    User Request: {standalone_question}

    Chat History:
    {chat_history}

    Error Details:
    {dataframe}
    """
        response = llm.invoke(error_prompt)
        return response.content.strip()

    elif dataframe.empty:   
    # Handle empty results
        no_data_prompt = f"""
    The following request was made, but the system could not find any relevant information. 
    Provide a concise and direct response to the user, following these special instructions:

    {special_instructions}

    User Request: {standalone_question}

    Chat History:
    {chat_history}
    """
        response = llm.invoke(no_data_prompt)
        return response.content.strip()

    else:
        # Convert the DataFrame to a comma-separated format including column names
        result_text = ", ".join(dataframe.columns) + "\n"  # Add column names
        result_text += "\n".join(dataframe.astype(str).apply(lambda row: ", ".join(row), axis=1).tolist())

        # Use the LLM to generate a response based on the query result and chat history
        response_prompt = f"""
Provide a response based on the retrieved information. Follow these special instructions:

{special_instructions}

User Request: {standalone_question}

Information Retrieved:
{result_text}

Chat History:
{chat_history}
"""
        response = llm.invoke(response_prompt)
        return response.content.strip()








#############################################################################

