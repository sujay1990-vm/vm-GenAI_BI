from flask import Flask, request, jsonify

import pandas as pd
import cohere

from auxiliary_functionalities import *

from quest_and_ans_prompt_template import *

from api_key import *

app = Flask(__name__)


llm = get_llm_chat_model()
embedding_model = get_embedding()
print("Initialised LLM chat model")

db_file = "claims.db"
table_name = "claims"
load_csv_to_sqlite(csv_file, db_file, table_name)
table_schema = generate_table_schema()
table_metadata = create_table_metadata()
faiss_store = create_faiss_store(csv_file, embedding_model)

@app.route("/",methods=['GET'])
def get_welcome_msg():
    return "Welcome to LLM Q&A Application"

# Initialize the Conversational RAG system

conv_manager = ConversationManager(history_limit=5)

# Example of handling the API request for Q&A
@app.route("/get_answer", methods=['POST'])
def get_answer():
    try:
        # Get the query from the request body
        data = request.json
        query_text = data.get('query', None)

        if not query_text:
            return jsonify({"error": "No query text provided"}), 400

        # Ask the question using the ConversationalRAG system
        standalone_question = conv_manager.reformulate_question(query_text, llm)
        sql_query = generate_sql_query(llm, standalone_question, table_metadata, table_name, table_schema)
        query_result = execute_sql_query_with_preprocessing(db_file, sql_query, faiss_store, embedding_model)
        response = generate_response(llm, query_result,standalone_question, conv_manager.get_conversation_context())
        conv_manager.update_conversation_history(standalone_question, response)
        # Return the response from the LLM model
        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

app.run(debug=False, host='0.0.0.0', port = 5000)
