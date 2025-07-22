import sqlite3
import os
import pandas as pd
from tabulate import tabulate

# For executing SQL queries, assume we have a function to execute against your SQLite DB.
def execute_query_node(state: dict) -> dict:
    # st.write("Executing SQL queries...")
    sql_queries = state.get("sql_queries", [])
    # Select DB filename based on domain
    db_filename = os.path.join(os.path.dirname(__file__), "merged.db")

    result_strs = []
    sql_results = []
    execution_error = False
    df_list = []

    if not sql_queries:
        error_msg = "No SQL queries found."
        state["sql_result_str"] = error_msg
        state["sql_result_df"] = []
        state["sql_results"] = []
        state["execution_error"] = True
        state.setdefault("error_history", []).append(error_msg)
        return state
    
    # For demonstration, we execute against our local SQLite DB (census.db)
    # db_filename = "census.db"
    print(sql_queries)
    conn = sqlite3.connect(db_filename)
    for i, query in enumerate(sql_queries):
        try:
            metric_name = f"Query {i+1}"  # or use resolved metric name if available
            df = pd.read_sql_query(query, conn)
            df_list.append(df)

            if df.empty:
                result_text = "(no results)"
                table_str = f"Query Result for '{metric_name}': No rows returned."
            else:
                result_text = tabulate(df, headers="keys", tablefmt="grid")
                table_str = f"Query Result for '{metric_name}':\n{result_text}"

            sql_results.append({"metric": metric_name, "result": result_text})
            result_strs.append(table_str)

        except Exception as e:
            metric_name = f"Query {i+1}"
            err_msg = str(e)
            table_str = f"❌ Error executing SQL for '{metric_name}':\n{query}\nException: {err_msg}"
            sql_results.append({"metric": metric_name, "result": f"ERROR: {err_msg}"})
            result_strs.append(table_str)
            execution_error = True
            df_list.append(pd.DataFrame())  # keep consistent length
            state.setdefault("error_history", []).append(table_str)

    conn.close()

    # Combine all results for state
    state["sql_result_str"] = "\n\n".join(result_strs)
    # Flatten all result DataFrames into records (can choose first or concatenate all)
    state["sql_result_df"] = pd.concat(df_list, ignore_index=True).to_dict(orient="records")
    state["sql_results"] = sql_results
    state["execution_error"] = execution_error

    return state