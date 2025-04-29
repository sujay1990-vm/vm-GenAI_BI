import streamlit as st
from langchain.agents import AgentExecutor
import textwrap
from agent_setup import *


# Streamlit App
st.title("AI Assistant 💼")
st.write("Ask your questions for personalized product recommendations or data insights.")

# Initialize chat history in session_state

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# User Input
user_query = st.text_input("Enter your query:", placeholder="e.g., What do you recommend for David Brown")

# When user submits query
if st.button("Answer") and user_query:
    with st.spinner("Thinking..."):
        try:
            # Invoke the agent with current chat history
            response = agent_executor.invoke({
                "input": user_query
            })

            # Extract response text safely
            output_text = response.get('output', '')

            # Clean the formatting (assuming fix_vertical_text function exists)
            # cleaned_response = fix_vertical_text(output_text)
            cleaned_response = format_recommendation_summary(output_text)
            
            # Store in session state
            st.session_state.last_response = cleaned_response

            # # Display the response
            # st.subheader("Answer:")
            # st.write(cleaned_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Enter a query and click 'Answer' to see results.")

# Always display last response if available
if st.session_state.last_response:
    st.subheader("Answer:")
    st.write(st.session_state.last_response)

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.last_response = ""
    st.success("Chat history cleared.")