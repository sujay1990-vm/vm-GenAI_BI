import pandas as pd
import os
import sys
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from prompts.census_prompts import *
from prompts.quality_prompts import *


def metadata_loader_node(state: dict) -> dict:
    """
    Loads table metadata and entity relationships for resolved domains.
    Assumes all domain prompt strings are already imported via config.py.
    """
    print("---METADATA LOADER NODE---")

    metadata_store = {}
    resolved_metrics = state.get("resolved_metrics", [])

    # Extract unique domain names from resolved metrics
    domains = {m["domain_name"].lower().replace(" ", "_") for m in resolved_metrics if m.get("domain_name")}
    print(domains)
    for domain in domains:
        table_var = f"{domain}_table_metadata"
        er_var = f"{domain}_entity_relationships"
        inst_var = f"{domain}_domain_instructions"
        sql_var = f"{domain}_sample_sql_queries"

        try:
            table_metadata = globals().get(table_var, "").strip()
            er_metadata = globals().get(er_var, "").strip()
            inst_metadata = globals().get(inst_var, "").strip()
            sql_metadata = globals().get(sql_var, "").strip()

            if not table_metadata:
                print(f"⚠️ Missing table metadata for domain: {domain}")
            if not er_metadata:
                print(f"⚠️ Missing entity relationships for domain: {domain}")
            if not inst_metadata:
                print(f"⚠️ Missing domain instructions for domain: {domain}")
            if not sql_metadata:
                print(f"⚠️ Missing sample sql queries for domain: {domain}")

            # ✅ Flattened prompt built immediately (no second loop later)
            flattened_prompt = f"""
                === DOMAIN: {domain.replace("_", " ").title()} ===

                TABLE METADATA:
                {table_metadata}

                ENTITY RELATIONSHIPS:
                {er_metadata}

                DOMAIN INSTRUCTIONS:
                {inst_metadata}

                SAMPLE SQL QUERIES:
                {sql_metadata}
                """.strip()

            metadata_store[domain] = {
                "table_metadata": table_metadata,
                "entity_relationships": er_metadata,
                "domain_instructions": inst_metadata,
                "sample_sql_queries": sql_metadata,
                "flattened_prompt": flattened_prompt  # ✅ flatten now
            }

            if all([table_metadata, er_metadata, inst_metadata, sql_metadata]):
                print(f"✅ Loaded metadata for domain: {domain}")

        except Exception as e:
            print(f"❌ Error loading metadata for domain {domain}: {e}")

    state["metadata"] = metadata_store
    return state