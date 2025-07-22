
from .llm import get_llm
import pandas as pd
import os
import sys
from pydantic import BaseModel, Field
from langchain_core.prompts import  ChatPromptTemplate
from typing import List, Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt
import io
import warnings

llm = get_llm()


# Define the VisualizationGoals model
def summarize_df(df):
    """
    Given a Pandas DataFrame, return a short summary string:
    e.g., "Columns: col1, col2, col3; Rows: 100"
    """
    cols = ", ".join(df.columns)
    return f"Columns: {cols}; Rows: {df.shape[0]}"

def get_sql_result_summary(state: dict) -> str:
    """
    Extracts a summary string from the list of DataFrames stored in state["sql_result_df"].
    If multiple DataFrames exist, summaries are joined with newlines.
    """
    records = state.get("sql_result_df", [])
    if not records:
        return "No SQL results available."

    df = pd.DataFrame(records)

    if df.empty:
        return "SQL result is an empty DataFrame."

    return f"Query Result Summary:\n{summarize_df(df)}"


class VisualizationGoals(BaseModel):
    """
    Represents the visualization goals to be achieved by the visuals.
    Exactly 3 concise and actionable goals should be returned.
    """
    goals: List[str] = Field(
        default_factory=list,
        description="A list of exactly 3 visualization goals to guide the creation of visuals."
    )


# Create a structured LLM that outputs a VisualizationGoals JSON object
structured_llm_visualization = llm.with_structured_output(schema=VisualizationGoals, method='function_calling')

# Create a prompt template for visualization goals
visualization_system = """\
You are a data visualization expert. Based on the user's query, the detected intent, the provided visualization instructions, and a summary of the SQL query results, generate exactly 3 concise visualization goals that the visuals should achieve.
Return a JSON object in the following format:
{{
  "goals": [
    "Visualization goal 1",
    "Visualization goal 2",
    "Visualization goal 3"
  ]
}}
"""

visualization_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", visualization_system),
        ("human", "User query: {user_query}\nVisualization instructions: {visual_instructions}\nSQL Result Summary: {sql_result_summary}")
    ]
)

# Combine the prompt and the structured LLM
visualization_detector = visualization_prompt | structured_llm_visualization

def identify_visualization_goals(user_query: str, visual_instructions: str, sql_result_summary: str) -> VisualizationGoals:
    """
    Generates visualization goals based on:
      - user_query: the original query text
      - visual_instructions: any explicit instructions the user provided for visualization
      - sql_result_summary: a brief summary of the SQL query results (e.g. key columns, row counts)
    Returns an instance of VisualizationGoals.
    """
    inputs = {
        "user_query": user_query,
        "visual_instructions": visual_instructions,
        "sql_result_summary": sql_result_summary
    }
    result = visualization_detector.invoke(inputs)
    return result


def identify_visualization_goals_from_state(state: dict):
    """
    Extracts inputs from the state (user_query, intent, visual_instructions, and a summary of the SQL result DataFrames)
    and passes them to the visualization goal detection chain.
    
    This function uses your existing 'identify_visualization_goals' function (which expects:
      user_query, intent, visual_instructions, sql_result_summary)
    and returns the resulting VisualizationGoals object.
    """
    # Extract the SQL result summary from the stored DataFrames.
    sql_result_summary = get_sql_result_summary(state)
    
    # Extract the other required state values.
    user_query = state.get("user_query", "")
    visual_instructions = state.get("visual_instructions", "")
    
    # Now call the visualization goal detection chain.
    # (Assuming you have defined the function 'identify_visualization_goals'
    # as in the previous example.)
    viz_goals = identify_visualization_goals(user_query, visual_instructions, sql_result_summary)
    state["goals"] = viz_goals.goals
    return state


######### NODE #####################

def visualization_generation_node(state: dict) -> dict:
    """
    Uses the SQL result DataFrames (state["sql_result_df"]), visualization goals (state["goals"]),
    intent, and visual_instructions to generate Python code (using seaborn) that creates visuals.
    The generated code is executed in an environment where the DataFrame list is available.
    Visuals are saved to a designated output directory, and the output path is stored in state["visualization_output"].
    """
    print("---VISUALIZATION GENERATION NODE---")
    # st.write("Creating visuals...")
    # Prepare a summary of the SQL result DataFrames.
    def summarize_df(df):
        cols = ", ".join(df.columns)
        return f"Columns: {cols}; Rows: {df.shape[0]}"
    
    records = state.get("sql_result_df", [])
    df = pd.DataFrame(records)
    sql_summary = summarize_df(df) if not df.empty else "Empty SQL result."
    
    # Extract necessary state values.
    user_query = state.get("user_query", "")
    visual_instructions = state.get("visual_instructions", "No specific instructions provided.")
    goals = state.get("goals", [])
    
    # Create a prompt for visualization code generation.
    prompt = f"""
You are a data visualization expert. Based on the following inputs, generate Python code using the seaborn library that produces visuals.
The code should define a function named create_visuals(df) that takes Pandas DataFrame as input and creates visuals.
Do not save any files to disk; simply create and display the figures.
Inputs:
User Query: {user_query}
Visualization Goals:
- {goals[0] if len(goals) > 0 else "N/A"}
- {goals[1] if len(goals) > 1 else "N/A"}
- {goals[2] if len(goals) > 2 else "N/A"}
Visualization Instructions: {visual_instructions}
    1. Always show count on the charts and visuals
    2. Use color coded visuals if possible
SQL Results Summary:
{sql_summary}

The generated code should:
1. Import seaborn (and matplotlib as needed).
2. Create visuals for the provided DataFrame.
3. Ensure your code starts with:
    import seaborn as sns
    import matplotlib.pyplot as plt
4. Output only valid Python code.
"""
    # Call the LLM to generate the visualization code.
    response_message = llm.invoke(prompt)
    visualization_code = response_message.content.strip()
    if visualization_code.startswith("```python"):
        visualization_code = visualization_code[len("```python"):].strip()
    elif visualization_code.startswith("```"):
        visualization_code = visualization_code[3:].strip()
    if visualization_code.endswith("```"):
        visualization_code = visualization_code[:-3].strip()
    
    # print("Generated visualization code:")
    # print(visualization_code)
    
    # 6. Save the generated code to a file.
    # Store the generated code in state for debugging if needed.
    state["generated_visualization_code"] = visualization_code
    
    # 7. Prepare an execution environment where df_list is available.
    exec_globals = {
        "__name__": "__main__",
        "os": os,
        "pd": pd,
        "sns": __import__("seaborn"),
        "plt": __import__("matplotlib.pyplot"),
        "df": df
    }
  
    try:
        exec(visualization_code, exec_globals, exec_globals)
        if "create_visuals" in exec_globals and callable(exec_globals["create_visuals"]):
            exec_globals["create_visuals"](df)
            figures = [plt.figure(num) for num in plt.get_fignums()]
            visualization_files = []
            for fig in figures:
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                visualization_files.append(buf.getvalue())
                plt.close(fig)
            state["visualization_files"] = visualization_files
            state["visualization_output"] = "Visualizations captured in memory."
            print("✅ Visualizations created and captured in memory.")
        else:
            state["visualization_output"] = "Visualization function not defined."
            print("❌ 'create_visuals' function not found.")
    except Exception as e:
        error_msg = f"Error executing visualization code: {e}"
        print(error_msg)
        state["visualization_output"] = error_msg
        state.setdefault("error_history", []).append(error_msg)

    state["visualization_response"] = f"Visualization process complete. {state['visualization_output']}"
    return state