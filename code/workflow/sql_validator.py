from typing import List, Dict, Any
import re


from collections import Counter

def extract_fact_tables(sql: str, fact_table_set: set) -> set:
    """
    Extracts fact tables used in the final SELECT query block (excluding CTEs).
    Case-insensitive match. Returns matched table names in original case.
    """
    lower_sql = sql.lower()
    if "with" in lower_sql:
        select_indices = [m.start() for m in re.finditer(r'\bselect\b', lower_sql)]
        if select_indices:
            sql = sql[select_indices[-1]:]

    pattern = r"\b(from|join)\s+([\w\.]+)"
    matches = re.findall(pattern, sql, re.IGNORECASE)

    matched = set()
    normalized_fact_tables = {t.lower() for t in fact_table_set}

    for _, table in matches:
        table_cleaned = table.strip().lower()
        if table_cleaned in normalized_fact_tables:
            matched.add(table_cleaned)

    print("🔍 Fact tables in final SELECT block:", matched)
    return matched


def sql_query_validation_node(state: dict) -> dict:
    print("🔍 Entering SQL Validator Node...")

    forbidden_patterns = [
        r"SUM\s*\(\s*(\w+\s*\.\s*)?ResidentKey\s*\)",
        r"SUM\s*\(\s*(\w+\s*\.\s*)?EmployeeID\s*\)",
        r"SUM\s*\(\s*(\w+\s*\.\s*)?[^)\s]*ID\s*\)",
        r"AVG\s*\(\s*(\w+\s*\.\s*)?[^)\s]*ID\s*\)"
    ]

    fact_tables = {
        "parker_uat_dataplatform.census_fact.Fact_AnnualCapacityAndBudget",
        "parker_uat_dataplatform.census_fact.Fact_Census",
        "parker_uat_dataplatform.hr_fact.fact_employeeleaverequestdetail",
        "parker_uat_dataplatform.hr_fact.fact_employeepayroll",
        "parker_uat_dataplatform.hr_fact.fact_employeesnapshot",
        "parker_uat_dataplatform.hr_fact.Fact_EmployeeTimeTracking",
        "parker_uat_dataplatform.hr_fact.Fact_EmployeeWorkSchedule",
        "parker_uat_dataplatform.hr_fact.fact_jobvacancysnapshot",
        "parker_uat_dataplatform.ehr_fact.Fact_MedicalEvent",
        "parker_uat_dataplatform.ehr_fact.Fact_HospitalTransfer",
        "parker_uat_dataplatform.common_fact.Fact_PhysicianOrder"
    }

    sql_queries = state.get("sql_queries", [])
    sql = "\n".join(sql_queries)
    state.setdefault("error_history", [])
    
    # ✅ Detect CTE presence
    if "with" in sql.lower():
        state["cte_flag"] = True
        state["cte_disclaimer"] = (
            "⚠️ This query involves multiple steps using WITH blocks. "
            "If the output looks too complex or inaccurate, consider simplifying your question "
            "or splitting it into smaller parts."
        )
    else:
        state["cte_flag"] = False
        state["cte_disclaimer"] = ""
        
    # Check for forbidden aggregation
    agg_violation = any(re.search(p, sql, re.IGNORECASE) for p in forbidden_patterns)
    if agg_violation:
        print("⚠️ SUM/AVG on ID column detected.")
        state["query_check_flag"] = True
        state["query_check_msg"] = "⚠️ SUM/AVG applied to ID column."
        state["error_history"].append(f"❌ SUM/AVG applied to ID column:\n{sql}")

    # Check for fact-to-fact join
    joined_facts = extract_fact_tables(sql, fact_tables)
    if len(joined_facts) > 1:
        print(f"🛑 Multiple fact tables joined: {joined_facts}")
        state["fact_join_violation"] = True
        state["fact_join_msg"] = (
            f"❌ Multiple fact tables joined in final query: {', '.join(joined_facts)}\n"
            "This may lead to data duplication or inflated results.\n"
            "Try breaking your question into simpler parts and combine the answers manually."
        )
        state["error_history"].append(state["fact_join_msg"])
    else:
        state["fact_join_violation"] = False
        state["fact_join_msg"] = ""

    # 🚫 Check for forbidden payroll fact table usage
    payroll_table = "parker_uat_dataplatform.hr_fact.fact_employeepayroll".lower()
    if payroll_table in joined_facts:
        print(f"🛑 Forbidden table accessed: {payroll_table}")
        state["hard_exit"] = True
        state["exit_reason"] = (
            f"❌ Query references a restricted table: `{payroll_table}`. "
            "Access to payroll-level data is not allowed via this interface."
        )
        state["error_history"].append(state["exit_reason"])
        return state  # ⛔️ Hard exit

    # ✅ Clean pass
    if not agg_violation and not state["fact_join_violation"]:
        print("✅ No SQL pattern violations found.")
        state["query_check_flag"] = False
        state["query_check_msg"] = ""

    return state


