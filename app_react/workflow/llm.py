from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import streamlit as st

def get_llm():
    return AzureChatOpenAI(
        temperature=0,
        deployment_name=st.secrets["OPENAI_DEPLOYMENT_NAME"],
        model_name=st.secrets["OPENAI_MODEL_NAME"],
        azure_endpoint=st.secrets["OPENAI_DEPLOYMENT_ENDPOINT"],
        openai_api_version=st.secrets["OPENAI_API_VERSION"],
        openai_api_key=st.secrets["OPENAI_API_KEY"],
    )

def get_embedding_model():
    return AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        azure_endpoint=st.secrets["OPENAI_DEPLOYMENT_ENDPOINT"],
        openai_api_version=st.secrets["embedding_api_version"],
        openai_api_key=st.secrets["OPENAI_API_KEY"],
    )
