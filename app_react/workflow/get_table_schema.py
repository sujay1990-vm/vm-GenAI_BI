from langchain_core.tools import tool
import json

@tool
def get_schema_table_tool(query: str) -> str:
    """
    Loads only the most relevant table schemas (top 3) based on query.
    Optimized for minimal token usage.
    """
    print(f"🔍 Finding relevant tables for: {query}")
    
    try:
        with open("table.json", "r") as f:
            table_descriptions = json.load(f)
        with open("new_schema.json", "r") as f:
            full_schema = json.load(f)
        with open("vocab_dictionary.json", "r") as f:
            metric_definitions = json.load(f)
    except FileNotFoundError as e:
        return f"Error: {e}"
    
    # Quick keyword matching
    query_lower = query.lower()
    scores = {}
    
    for table_name, info in table_descriptions.items():
        score = 0
        text = f"{table_name} {info['description']}".lower()
        
        # Score based on query keywords
        for word in query_lower.split():
            if len(word) > 2:
                score += text.count(word) * 2
        
        # Bonus for exact table name match
        if table_name.replace('_', ' ') in query_lower:
            score += 20
        
        scores[table_name] = score
    
    # Get top 3
    top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    selected_tables = [name for name, _ in top_3]
    
    print(f"✅ Selected: {', '.join(selected_tables)}")
    
    # Build minimal response
    result = ["=== TOP 3 RELEVANT TABLES ===\n"]
    
    for table_name in selected_tables:
        schema_key = f"{table_name}_data" if f"{table_name}_data" in full_schema else table_name
        
        if schema_key in full_schema:
            result.append(f"\n{table_name}:")
            # Only include essential schema info
            result.append(json.dumps(full_schema[schema_key], indent=2))
    
    # Only include relevant metrics (optional - can remove entirely)
    # result.append("\n\n=== METRICS ===")
    # result.append(json.dumps(metric_definitions, indent=2))
    
    return "\n".join(result)