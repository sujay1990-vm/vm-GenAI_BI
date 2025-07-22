from typing import Literal, Dict, Any
import re


def handle_metric_resolution(state: dict) -> Literal["MetadataLoader", "final_output"]:
    if state.get("flow_exit_flag"):
        return "final_output"
    return "MetadataLoader"


def handle_query_checks(state: dict) -> Literal["SQL_generator", "sql_executor", "final_output"]:
    # 🛑 Global exit flag (e.g., retry exhaustion, domain failure)
    if state.get("flow_exit_flag", False):
        print(f"🛑 Early exit triggered — Reason: {state.get('exit_reason')}")
        return "final_output"    
    
    # 🛑 Hard exit check (e.g., forbidden table like payroll)
    if state.get("hard_exit", False):
        print(f"🛑 Hard exit triggered — Reason: {state.get('exit_reason')}")
        return "final_output"

    # ❌ Fatal error: Fact-to-fact join
    if state.get("fact_join_violation", False):
        print("🛑 Fatal: Fact-to-fact join detected — exiting.")
        state["fact_join_msg"] = "❌ Exiting: Multiple fact tables joined. This may lead to inflated results or exploding joins."
        return "final_output"

    # ♻️ Retryable error: Fixable issues (like SUM on ID)
    if state.get("query_check_flag", False):
        print("♻️ Fixable issue detected — regenerating SQL.")
        return "SQL_generator"

    # ✅ Pass-through
    print("✅ SQL passed validation — proceeding to execution.")
    return "sql_executor"



def handle_execution_result(state: dict) -> Literal["SQL_generator", "final_output"]:
    """
    Determines the next step based on SQL execution outcome:
    - Retry SQL generation if it's a non-permission error.
    - Exit if it's a permission error.
    - Proceed to final output if no error occurred.
    """
    
    if state.get("execution_error", False):
        last_err = ""
        if state.get("error_history"):
            last_err = state["error_history"][-1]

        # Retry SQL generation for other errors
        return "SQL_generator"

    # Proceed if no errors
    return "final_output"
