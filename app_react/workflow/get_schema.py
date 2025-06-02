from langchain_core.tools import tool
import json

@tool
def get_schema_tool(_: str = "") -> str:
    """
    Loads the full table schema and metric definitions for SQL generation and reasoning.
    Returns a string summary combining both files.
    """
    print("📄 Loading schema and metric definitions...")

    try:
        with open("new_schema.json", "r") as f:
            table_schema = json.load(f)

        with open("vocab_dictionary.json", "r") as f:
            metric_definitions = json.load(f)

    except FileNotFoundError as e:
        return f"Error loading schema: {e}"

    # Optional: format output as readable text
    schema_str = json.dumps(table_schema, indent=2)
    metrics_str = json.dumps(metric_definitions, indent=2)

    combined_summary = f"""=== TABLE SCHEMA ===\n{schema_str}\n\n=== METRIC DEFINITIONS ===\n{metrics_str}"""

    return combined_summary