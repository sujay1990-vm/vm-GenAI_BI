import pandas as pd
import datetime
from typing import List, Dict, Any
import io
import re



def report_generation_node(state: dict) -> dict:
    """
    Generates in-memory CSV reports from DataFrames in state["sql_result_df"].
    Stores result in state["csv_files"] as {filename: bytes}.
    """
    import datetime
    import io

    print("📄 Starting Report Generation...")

    records = state.get("sql_result_df", [])
    df = pd.DataFrame(records)
    resolved_metrics = state.get("resolved_metrics", [])
    csv_files = {}

    try:
        if df.empty:
            print("⚠️ Skipping empty DataFrame.")
        else:
            raw_name = resolved_metrics[0].get("metric", "query_1") if resolved_metrics else "query_1"
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '', raw_name.replace(" ", "_")).lower()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{safe_name}_{timestamp}.csv"

            csv_content = df.to_csv(index=False)
            csv_files[filename] = csv_content.encode("utf-8")

            print(f"✅ Query written to memory: {filename}")

        state["csv_files"] = csv_files
        state["report_response"] = "Report generated successfully."
        state["report_generation_error"] = False

    except Exception as e:
        error_msg = f"❌ Critical error in report generation: {e}"
        print(error_msg)
        state.setdefault("error_history", []).append(error_msg)
        state["report_generation_error"] = True
        state["report_response"] = "Report generation failed."

    return state