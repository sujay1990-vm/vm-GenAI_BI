from langgraph.constants import Send


def assign_workers(state: dict):
    """Route to either RAG or SQL worker based on subquery intent"""
    return [
        Send("rag_worker", {"query": sq.query}) if sq.intent == "rag"
        else Send("sql_worker", {"query": sq.query})
        for sq in state["subqueries"]
    ]