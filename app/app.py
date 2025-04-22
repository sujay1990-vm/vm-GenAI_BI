import streamlit as st
from langchain.agents import AgentExecutor
import textwrap
from agent_setup import *


# Streamlit App
st.title("AI Assistant 💼")
st.write("Ask your questions for personalized product recommendations or data insights.")

# Initialize chat history in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
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
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })

            # Extract response text safely
            output_text = response.get('output', '') if isinstance(response, dict) else getattr(response, 'content', str(response))

            # Clean the formatting (assuming fix_vertical_text function exists)
            cleaned_response = fix_vertical_text(output_text)
            
            # Store in session state
            st.session_state.last_response = cleaned_response

            # # Display the response
            # st.subheader("Answer:")
            # st.write(cleaned_response)

            # Update chat history
            from langchain_core.messages import AIMessage, HumanMessage
            st.session_state.chat_history.extend([
                HumanMessage(content=user_query),
                AIMessage(content=output_text),
            ])

            # Limit chat history to last 6 messages (3 exchanges)
            if len(st.session_state.chat_history) > 6:
                st.session_state.chat_history = st.session_state.chat_history[-6:]

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Enter a query and click 'Answer' to see results.")

# Always display last response if available
if st.session_state.last_response:
    st.subheader("Answer:")
    st.write(st.session_state.last_response)

# Show chat history toggle
if st.checkbox("Show Conversation History"):
    for msg in st.session_state.chat_history:
        role = "You" if isinstance(msg, HumanMessage) else "AI"
        st.markdown(f"**{role}:** {msg.content}")

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.last_response = ""
    st.success("Chat history cleared.")