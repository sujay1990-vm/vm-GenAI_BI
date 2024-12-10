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

<Special Instructions/>

Convert the following natural language query into a SQL query:

{natural_language_query}

Only return the SQL query, no additional text.
"""
    response = llm.invoke(query_prompt)
    return response.content

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
"""
    
    if dataframe.empty:
        # Use the LLM to generate a specific response for no data
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
