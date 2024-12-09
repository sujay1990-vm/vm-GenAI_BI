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
- Summarize or Explain means to read the entire Unstructured Column requested

<Special Instructions/>

Convert the following natural language query into a SQL query:

{natural_language_query}

Only return the SQL query, no additional text.
"""
    response = llm.invoke(query_prompt)
    return response.content

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
        # Convert the DataFrame to a comma-separated format including column names
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
