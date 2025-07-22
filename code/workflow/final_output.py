from .llm import get_llm
from .nl_response import nl_response_node
from .visualization import identify_visualization_goals_from_state
from .visualization import visualization_generation_node
from .report import report_generation_node
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

llm = get_llm()

def final_output_node(state: dict) -> dict:
    """
    Final output node that determines what to generate based on user intent:
    - If report is requested → run report_generation_node
    - Else if visualization is requested → run nl_response_node and visualization_generation_node
    - Else → run only nl_response_node
    Stores the result in state["final_response"]
    """
    print("---FINAL OUTPUT NODE---")

    if state.get("flow_exit_flag"):
        reason = state.get("exit_reason", "").lower()

        if "domain" in reason:
            print("🛑 Final output skipped due to unresolved or vague domain.")
            state["final_response"] = state.get(
                "final_response",
                "❌ Your question was too vague for me to determine a specific metric or domain. Please rephrase it with a clear metric, time frame, and optional location."
            )

        elif "retries" in reason or "SQL generation failed" in reason:
            print("🛑 Final output skipped due to repeated SQL generation failures.")
            state["final_response"] = state.get(
                "final_response",
                "❌ We couldn't generate a working query despite multiple attempts. Try rephrasing your question more clearly, or break it into smaller parts."
            )

        else:
            print("🛑 Final output skipped — unspecified exit reason.")
            state["final_response"] = state.get("final_response", "❌ Unable to continue due to an unknown issue.")

        return state




    # 🛑 Exit immediately if hard exit was triggered
    if state.get("hard_exit", False):
        print("🛑 Hard exit triggered — exiting with message.")
        state["final_response"] = state.get(
            "exit_reason",
            "❌ Your query was blocked because it referenced a restricted dataset (e.g., payroll). This data is not available via this interface."
        )
        return state
    
    # 🛑 Stop if fact-fact join violation detected
    if state.get("fact_join_violation", False):
        print("🛑 Final output aborted due to fatal join issue.")
        state["final_response"] = state.get(
            "fact_join_msg",
            "❌ The query attempted to join multiple large data tables directly, which can lead to duplicated rows and incorrect results.\n\n"
            "➡️ Try breaking your question into smaller, more specific parts. For example:\n"
            "- Ask for one metric or calculation at a time\n"
            "- Then ask separately for the next related metric\n\n"
            "This helps ensure accurate results and prevents performance or data quality issues."
        )
        return state

    # 🧠 Generate based on user intent
    if state.get("report", False):
        state = report_generation_node(state)
        final_response = state.get("report_response", "No report generated.")

    elif state.get("visualize", False):
        state = identify_visualization_goals_from_state(state)
        state = visualization_generation_node(state)
        final_response = state.get("visualization_response", "No visualization generated.")

    else:
        state = nl_response_node(state)
        final_response = state.get("nl_response", "No NL response found.")

    # # ⚠️ Append disclaimer if CTE was detected
    # if state.get("cte_flag", False):
    #     final_response += "\n\n" + state.get("cte_disclaimer", "")

    state["final_response"] = final_response
    return state

    
