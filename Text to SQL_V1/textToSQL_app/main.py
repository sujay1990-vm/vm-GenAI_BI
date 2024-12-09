from langchain_openai import AzureChatOpenAI
from config import *
from database import load_csv_to_sqlite, execute_sql_query
from metadata import create_table_metadata, get_custom_table_schema
from conversation_manager import ConversationManager
from llm_utils import generate_sql_query, generate_response

# Initialize LLM and Conversation Manager
llm = AzureChatOpenAI(
    temperature=0.1,
    deployment_name=OPENAI_DEPLOYMENT_NAME,
    model_name=OPENAI_MODEL_NAME,
    azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
    openai_api_version=OPENAI_API_VERSION,
    openai_api_key=OPENAI_API_KEY
)

conv_manager = ConversationManager(history_limit=3)

# Load CSV into SQLite
table_name = "nurse_notes"
load_csv_to_sqlite(CSV_FILE_PATH, DB_FILE_PATH, table_name)

# Generate metadata and schema
table_metadata = create_table_metadata()
table_schema = get_custom_table_schema()

# User Interaction
user_question = input("Enter your question: ")

standalone_question = conv_manager.reformulate_question(user_question, llm)
sql_query = generate_sql_query(llm, table_metadata, standalone_question, table_name, table_schema)
query_result = execute_sql_query(DB_FILE_PATH, sql_query)

if isinstance(query_result, str):
    print(query_result)
else:
    query_response = generate_response(llm, query_result, standalone_question, conv_manager.get_conversation_context())
    print(query_response)
    conv_manager.update_conversation_history(standalone_question, query_response)
