import streamlit as st
from langchain.agents import AgentExecutor
import textwrap
from agent_setup import agent_executor


# Streamlit App
st.title("AI Financial Advisor 💼")
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

            # Format the response for readability
            formatted_response = "\n".join([textwrap.fill(line, width=200) for line in output_text.split('\n')])

            # Display Result
            st.subheader("Recommendation:")
            st.markdown(f"```\n{formatted_response}\n```")


        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Enter a query and click 'Get Recommendation' to see results.")
