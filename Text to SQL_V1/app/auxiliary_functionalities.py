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
def num_tokens_from_string(string) -> int:
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    
    num_tokens = len(encoding.encode(string))

    return num_tokens

def read_csv_df(csv_path):
    return pd.read_csv(csv_path)

def add_num_tokens(df, column='Conversation',take_all_row_value = False):
    if take_all_row_value:
        df["total_tokens"] = num_tokens_from_string("\n".join([str(elem) if type(elem) != str else elem for elem in df.iloc[0].to_list()]))
    else:
        df["total_tokens"] = df[column].apply(num_tokens_from_string)
    return df

def read_filedata(filepath):
    with open(filepath) as file:
        data = file.read()
    return data

def save_to_txt(txt_filename,text_elem):
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"{text_elem}\n")
    print("Saved to text filename: ",txt_filename)

def get_current_time():
    print("In func-> get_current_time")
    currentTimeInIndia = datetime.datetime.strptime(
                                datetime.datetime.now(
                                                        pytz.timezone("Asia/Kolkata")
                                                    ).strftime("%d-%m-%Y %H:%M:%S"), "%d-%m-%Y %H:%M:%S"
                                                    ).strftime("%d-%m-%Y %I:%M %p")
    return currentTimeInIndia

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
                        temperature=0,
                        deployment_name=OPENAI_DEPLOYMENT_NAME,
                        model_name=OPENAI_MODEL_NAME,
                        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                        openai_api_version=OPENAI_API_VERSION,
                        openai_api_key=OPENAI_API_KEY               
                    )
    return llm_chat_model


#############################################################################


def load_csv_to_sqlite(csv_file, db_file, table_name):
    conn = sqlite3.connect(db_file)
    df = pd.read_csv(csv_file, encoding='latin-1')
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Data from {csv_file} loaded into SQLite table {table_name}.")


# Define a custom schema
def get_custom_table_schema():
    schema = {
        "Primary_key": "INTEGER (Structured)",
        "First_Name": "TEXT (Structured)",
        "Last_Name": "TEXT (Structured)",
        "Age": "INTEGER (Structured)",
        "Gender": "TEXT (Structured)",
        "Marital_Status": "TEXT (Structured)",
        "Race": "TEXT (Structured)",
        "Day": "TEXT (Structured)",
        "Shift": "TEXT (Structured)",
        "Nurse_name": "TEXT (Structured)",
        "Clinical_Notes": "TEXT (Unstructured)",
        "Non_clinical_notes": "TEXT (Unstructured)"
    }
    return "\n".join([f"{col}: {desc}" for col, desc in schema.items()])


import pandas as pd

def create_table_metadata():
    """
    Create a DataFrame containing metadata for a table.

    Returns:
        pd.DataFrame: A DataFrame with metadata, including column names, types, and descriptions.
    """
    metadata = {
        "Column_Name": [
            "Primary_key", "First_Name", "Last_Name", "Age", "Gender",
            "Marital_Status", "Race", "Day", "Shift", "Nurse_name",
            "Clinical_Notes", "Non_clinical_notes"
        ],
        "Column_Type": [
            "INTEGER", "varchar", "varchar", "INTEGER", "varchar",
            "varchar", "varchar", "varchar", "varchar", "varchar",
            "varchar", "varchar"
        ],
        "Description": [
            "Unique identifier for each record",
            "First name of the senior resident",
            "Last name of the senior resident",
            "Age of the senior resident",
            "Gender of the senior resident (Male/Female)",
            "Marital status of the resident (Single/Married/Widowed/Divorced)",
            "Race of the resident",
            "Day of the week the note was recorded",
            "Shift during which the note was recorded (Morning/Night)",
            "Full Name of the nurse who recorded the note",
            "Detailed clinical notes about the resident's incident or medical event/ emergency",
            "Non-clinical notes such as personal preferences or complaints"
    ]
    }
    return pd.DataFrame(metadata)


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
        self.history.append(f"Answer: {answer}")
        if len(self.history) > self.history_limit:
            self.history.pop(0)  # Remove the oldest entry to keep the history within the limit

    def get_conversation_context(self):
        """
        Get the full conversation context as a string.
        """
        return "\n\n".join(self.history)
    
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
    

# Generate SQL Query using AzureChatOpenAI LLM
def generate_sql_query(llm, natural_language_query, table_metadata, table_name, schema):
    query_prompt = f"""
You are a SQL expert. Use the table schema provided below to generate a SQL query. The data is regarding a senior care living community. 

<table info>

Table: {table_name}
Schema:
{schema}
Metadata: 
{table_metadata}

<table info/>

<Special Instructions>
- The SQL query must extract data semantically from text-based columns like 'Clinical_Notes' , 'Non_Clinical_Notes'
without relying on exact pattern matching (e.g., LIKE or CONTAINS).
- Use ( LIKE or CONTAINS ) when searching by Nurse name. e.g. WHERE Nurse_name LIKE "%<nurse name>%"
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
    Generate a natural language response based on the standalone question,
    the SQL query result, and the chat history. If no data is found, generate
    a response indicating the absence of relevant information in a specific manner,
    avoiding technical terms like 'database', 'query', or 'SQL'.
    """
    if dataframe.empty:
        # Use the LLM to generate a specific response for no data
        no_data_prompt = f"""
The following request was made, but the system could not find any relevant information. 
Provide a clear response to the user, avoiding technical terms like 'database', 'query', or 'SQL'. 
Focus on delivering an explanation in plain language.

User Request: {standalone_question}

Chat History:
{chat_history}
"""
        response = llm.invoke(no_data_prompt)
        return response.content.strip()
    else:
        # Convert the entire query result into a comma-separated string
        result_text = ", ".join(dataframe.columns) + "\n"  # Add column names
        result_text += "\n".join(dataframe.astype(str).apply(lambda row: ", ".join(row), axis=1).tolist())

        # Use the LLM to generate a response based on the query result and chat history
        response_prompt = f"""
Based on the information retrieved and the previous conversation context, provide a clear, concise explanation 
to address the user's request. Avoid using technical terms like 'database', 'query', or 'SQL'. 
Focus on explaining the information naturally and in plain language.

User Request: {standalone_question}

Information Retrieved:
{result_text}

<Special Instructions>:
- Always mention Resident's names whenever possible. Combine First Name and Last Name for Resident names

<Special Instructions/>

"""
        response = llm.invoke(response_prompt)
        return response.content.strip()




#############################################################################

