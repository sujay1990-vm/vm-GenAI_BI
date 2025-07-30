import streamlit as st
import sqlite3
import os
import re
import datetime
import random
import pandas as pd
import uuid 
import sys
# Import necessary components from your workflow code
from workflow.graph import app
import seaborn as sns
import matplotlib.pyplot as plt
import io
# Import your custom prompts (ensure your prompts.py is in the same directory)
from prompts import *
import warnings
from workflow.llm import get_llm, get_embedding_model
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

def generate_thread_id():
    return str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if st.button("🧹 Clear History"):
    st.session_state.thread_id = generate_thread_id()
    st.session_state.chat_history = []  # Or your own session keys
    st.success("✅ Started a new thread!")

llm = get_llm()
embeddings = get_embedding_model()

# --- Streamlit UI ---

def render_assistant_output(final_state, entry_index=0):
    final_output = final_state.get("final_response", "No response generated.")
    # cache_key = final_state.get("sql_cache_key")
    trace_id = final_state.get("mlflow_trace_id")
    render_id = f"{entry_index}"
    # ✅ 1. Display assistant response first
    st.markdown(final_output)
    unique_suffix = uuid.uuid4().hex
    # ✅ 2. Then feedback form — always comes immediately after the output

    # ✅ 3. Compute cache key for downstream (visuals, CSVs, etc)
    cache_key = f"auto_{hash(final_output)}"    

    # ✅ Expanders should always show, even after feedback
    if final_state.get("resolved_metrics"):
        st.markdown("<div style='font-size:20px; font-weight:bold;'>📊 Metrics & Definitions</div>", unsafe_allow_html=True)
        with st.expander("▼ Click to expand"):
            for metric in final_state["resolved_metrics"]:
                st.markdown(f"**Metric**: {metric.get('metric', 'N/A')}")
                st.markdown(f"**Definition**: {metric.get('definition', 'N/A')}")
                st.markdown(f"**Formula**: {metric.get('formula', 'N/A')}")
                st.markdown(f"**Domain**: {metric.get('domain_name', 'N/A')}")
                st.markdown("---")

    if final_state.get("sql_queries"):
        st.markdown("<div style='font-size:20px; font-weight:bold;'>🧠 SQL Query Executed</div>", unsafe_allow_html=True)
        with st.expander("▼ Click to expand"):
            st.code("\n".join(final_state["sql_queries"]), language="sql")

    if final_state.get("sql_result_str"):
        st.markdown("<div style='font-size:20px; font-weight:bold;'>🧾 SQL Result Summary</div>", unsafe_allow_html=True)
        with st.expander("▼ Click to expand"):
            st.text(final_state["sql_result_str"])

    if final_state.get("csv_files"):
        st.markdown("<div style='font-size:20px; font-weight:bold;'>📥 Download CSV Files</div>", unsafe_allow_html=True)
        with st.expander("▼ Click to expand"):
            for i, (filename, data) in enumerate(final_state["csv_files"].items()):
                st.download_button(
                    label=f"Download {filename}",
                    data=data,
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_csv_{render_id}_{unique_suffix}"
                )

    # ✅ Render visuals ONLY if this message actually has them
    if final_state.get("visualization_files"):
        st.subheader("🖼️ Visualizations")
        for i, img_bytes in enumerate(final_state["visualization_files"]):
            st.image(img_bytes, caption=f"Visualization {i+1}", use_column_width=True)

        st.markdown("<div style='font-size:20px; font-weight:bold;'>📸 Download Visuals</div>", unsafe_allow_html=True)
        with st.expander("▼ Click to expand"):
            for i, img_bytes in enumerate(final_state["visualization_files"]):
                st.download_button(
                    label=f"Download Visualization {i+1}",
                    data=img_bytes,
                    file_name=f"visualization_{i+1}.png",
                    mime="image/png",
                    key=f"download_visual_{render_id}_{unique_suffix}"
                )

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_user_prompt" not in st.session_state:
    st.session_state.pending_user_prompt = None

# if "feedback_given" not in st.session_state:
#     st.session_state.feedback_given = {}

# if "feedback_text" not in st.session_state:
#     st.session_state.feedback_text = ""


def main():
    st.markdown(
    """
    <style>
    /* Zoom everything by increasing the base font size */
    html, body, [class*="css"] {
        font-size: 22px !important;
    }

    /* Zoom in chat messages */
    .stChatMessage {
        font-size: 22px !important;
    }

    /* Zoom in chat input text */
    textarea {
        font-size: 22px !important;
    }

    /* Increase title size */
    h1 {
        font-size: 40px !important;
        font-weight: 800 !important;
    }

    .streamlit-expanderHeader {
            font-size: 24px !important;
            font-weight: bold !important;
            line-height: 1.6 !important;
            color: #ffffff !important;
        }

    .stExpander > summary {
            font-size: 24px !important;
            font-weight: 700 !important;
            line-height: 1.6 !important;
        }

    /* Optionally increase markdown block font too */
    .stMarkdown p {
        font-size: 22px !important;
    }
    </style>

    """,
    unsafe_allow_html=True,
    )

    st.title("Intelligence Assistant App")
    st.markdown("Ask your query below. Your conversation history will appear like a chat interface.")
    st.markdown('<h3 style="font-size:30px; font-weight:700;">💬 Sample Questions</h3>', unsafe_allow_html=True)
    all_questions = [
        "What is the Census for March 1st 2024 in Somerset?",
        "What is the daily Census for 1st of July 2024 in Somerset by Unit name",
        "Show a chart daily Census distribution for Somerset for March 2024",
        "are there anomalies in daily census for March 2024 for Somerset ?",
        "Summarize Clinical Notes for Ann Bell by date",
        "Create a report for Resident with events showing Name, Event name and count, date for Dec 2024 ?" ,
        "What is the most common prescription for abrasion injury",
        "Show number of wounds for 2024 by month"
    ]

    if "sample_questions" not in st.session_state:
        st.session_state.sample_questions = random.sample(all_questions, 4)

    for i, q in enumerate(st.session_state.sample_questions):
        if st.button(q, key=f"qbtn_{i}"):
            st.session_state.pending_user_prompt = q
            st.rerun()

    if st.button("🔄 Refresh Sample Questions", key="refresh_qs"):
        st.session_state.sample_questions = random.sample(all_questions, 4)
        st.rerun()

    # --- New Prompt ---
    user_prompt = st.chat_input("Ask your query...")
    if user_prompt:
        st.session_state.pending_user_prompt = user_prompt
        st.rerun()

    # --- Chat History ---
    for idx, entry in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(entry["user_query"])

        with st.chat_message("assistant"):
            render_assistant_output(entry["final_state"], entry_index=idx)


    # --- New Prompt Execution ---
    if st.session_state.get("pending_user_prompt"):
        prompt = st.session_state.pending_user_prompt
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                config = {
                    "configurable": {
                        "user_id": st.session_state.user_id,
                        "thread_id": st.session_state.thread_id
                    }
                }
                initial_state = {"user_query": prompt, "visualization_files": [], "csv_files": {}}
                final_state = app.invoke(initial_state, config=config)  
            render_assistant_output(final_state)

        st.session_state.chat_history.append({
            "user_query": prompt,
            "final_state": final_state
        })
        st.session_state.pending_user_prompt = None
        st.rerun()

if __name__ == "__main__":
    main()
