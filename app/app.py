import streamlit as st
from langchain.agents import AgentExecutor
import textwrap
from agent_setup import *


# Streamlit App
st.title("AI Assitant 💼")
st.write("Ask for personalized product recommendations based on customer profiles.")

# User Input
user_query = st.text_input("Enter your query:", placeholder="e.g., What do you recommend for David Brown")

# When user submits query
if st.button("Get Recommendation") and user_query:
    with st.spinner("Analyzing customer profile and generating recommendations..."):
        try:
            # Invoke the agent
            response = agent_executor.invoke({"input": user_query})

            # Extract response text safely
            output_text = response.get('output', '') if isinstance(response, dict) else getattr(response, 'content', str(response))

            # Clean the weird formatting
            cleaned_response = fix_vertical_text(output_text)

            st.subheader("Recommendation Summary:")
            st.write(cleaned_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Enter a query and click 'Get Recommendation' to see results.")
