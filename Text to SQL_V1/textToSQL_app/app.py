import streamlit as st
from config import *
from database import load_csv_to_sqlite, execute_sql_query
from metadata import create_table_metadata, get_custom_table_schema
from conversation_manager import ConversationManager
from llm_utils import generate_sql_query, generate_response
from langchain_openai import AzureChatOpenAI


st.set_page_config(page_title="Nurse Notes Query App", layout="wide")

# Initialize LLM
llm = AzureChatOpenAI(
    temperature=0.1,
    deployment_name=OPENAI_DEPLOYMENT_NAME,
    model_name=OPENAI_MODEL_NAME,
    azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
    openai_api_version=OPENAI_API_VERSION,
    openai_api_key=OPENAI_API_KEY
)

# Ensure all required session state variables are initialized
if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = ConversationManager(history_limit=5)

if "conversation" not in st.session_state:
    st.session_state.conversation = []  # Initialize conversation history

if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

if "table_metadata" not in st.session_state:
    st.session_state.table_metadata = create_table_metadata()

if "table_schema" not in st.session_state:
    st.session_state.table_schema = get_custom_table_schema()

if "user_input" not in st.session_state:
    st.session_state.user_input = ""  # For the user question input

if "sql_query" not in st.session_state:
    st.session_state.sql_query = None  # Persist the SQL query

if "query_result" not in st.session_state:
    st.session_state.query_result = None  # Persist the query result

if "assistant_response" not in st.session_state:
    st.session_state.assistant_response = None  # Persist the LLM response

# Database and Metadata Setup
table_name = "nurse_notes"
if not st.session_state.db_initialized:
    load_csv_to_sqlite(CSV_FILE_PATH, DB_FILE_PATH, table_name)
    st.session_state.db_initialized = True  # Mark as initialized

# App title and description
st.title("Nurse Notes Query App")
st.subheader("Interact with the nurse notes database using natural language queries.")

# Display conversation history at the top
st.markdown("### Conversation History")
chat_container = st.container()
with chat_container:
    # Make the conversation history scrollable
    st.write(
        """
        <style>
        .scrollable-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    scrollable_history = st.container()
    with scrollable_history:
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
        for item in st.session_state.conversation:
            st.markdown(f"**User:** {item['question']}")
            st.markdown(f"**Assistant:** {item['answer']}")
        st.markdown('</div>', unsafe_allow_html=True)

# Add input box at the bottom for typing new questions
st.markdown("### Your Query")
# Input box inside a form
with st.form("question_form", clear_on_submit=True):
    temp_input = st.text_input("Your question:", value="", key="temp_input", label_visibility="collapsed")
    submitted = st.form_submit_button("Submit")  # Submit on pressing Enter

# Trigger question submission
if submitted and temp_input:
    st.session_state.user_input = temp_input

    # Reformulate the question
    with st.spinner("Reformulating question..."):
        standalone_question = st.session_state.conversation_manager.reformulate_question(
            st.session_state.user_input, llm
        )
        st.session_state.standalone_question = standalone_question  # Store in session state

    # Generate SQL query
    with st.spinner("Generating SQL query..."):
        sql_query = generate_sql_query(
            llm,
            st.session_state.table_metadata,
            st.session_state.standalone_question,
            table_name,
            st.session_state.table_schema,
        )
        st.session_state.sql_query = sql_query  # Persist SQL query

    # Execute SQL query
    with st.spinner("Executing SQL query..."):
        query_result = execute_sql_query(DB_FILE_PATH, st.session_state.sql_query)
        if isinstance(query_result, str):  # SQL error
            st.session_state.assistant_response = query_result
        else:
            # Generate the LLM response
            with st.spinner("Generating response..."):
                assistant_response = generate_response(
                    llm,
                    query_result,
                    st.session_state.standalone_question,
                    st.session_state.conversation_manager.get_conversation_context(),
                )
                st.session_state.query_result = query_result  # Persist query result
                st.session_state.assistant_response = assistant_response  # Persist response

        # Save to conversation history
        st.session_state.conversation.append({
            "question": st.session_state.user_input,
            "answer": st.session_state.assistant_response,
        })

# Display persisted SQL query
if st.session_state.sql_query:
    st.markdown("### Generated SQL Query")
    st.code(st.session_state.sql_query, language="sql")

# Display LLM response
if st.session_state.assistant_response:
    st.markdown("### Assistant Response")
    st.success(st.session_state.assistant_response)

# Clear conversation history button
if st.button("Clear Chat History"):
    st.session_state.conversation_manager.clear_history()
    st.session_state.conversation = []
    st.session_state.sql_query = None
    st.session_state.query_result = None
    st.session_state.assistant_response = None
    st.success("Chat history cleared.")
