from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import streamlit as st
import os

OPENAI_DEPLOYMENT_ENDPOINT = "https://advancedanalyticsopenaikey.openai.azure.com/"
OPENAI_DEPLOYMENT_ENDPOINT_embed = "https://advancedanalyticsopenaikey.openai.azure.com/"
OPENAI_API_KEY = "REDACTED_API_KEY" 
OPENAI_API_VERSION = "2024-12-01-preview"
OPENAI_API_KEY_EMBEDDINGS = "REDACTED_API_KEY" 
OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"
OPENAI_MODEL_NAME="gpt-4o-mini"
embedding_api_version = "2024-02-01"

def get_llm():
    return AzureChatOpenAI(
                        temperature=0,
                        deployment_name=OPENAI_DEPLOYMENT_NAME,
                        model_name=OPENAI_MODEL_NAME,
                        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                        openai_api_version=OPENAI_API_VERSION,
                        openai_api_key=OPENAI_API_KEY            
                    )

def get_embedding_model():
    return AzureOpenAIEmbeddings(
                        deployment="text-embedding-3-small",
                        model="text-embedding-3-small",
                        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT_embed,
                        openai_api_version=embedding_api_version,
                        openai_api_key=OPENAI_API_KEY)
