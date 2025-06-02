import streamlit as st
import sys
from graph import build_graph, tool_usage_prompt, store
import streamlit as st
import os
import warnings
from datetime import datetime
import uuid
import time
from llm import get_llm, get_embedding_model
from rag_worker import retriever

llm = get_llm()
embeddings = get_embedding_model()

warnings.filterwarnings("ignore", category=FutureWarning)

# 🔐 Session Setup: User ID & Thread ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id  # ✅ Always set this outside the if block


def generate_thread_id():
    return str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

thread_id = st.session_state.thread_id


if st.button("🧹 Clear History"):
    st.session_state.thread_id = generate_thread_id()
    st.session_state.chat_history = []  # Or your own session keys
    st.success("✅ Started a new thread!")

# with st.sidebar:
#     with st.expander("⚙️ Developer Options", expanded=False):
#         show_ids = st.checkbox("Show Session Info", value=False)
#         if show_ids:
#             st.markdown(f"👤 **User ID**: `{st.session_state.user_id}`")
#             st.markdown(f"🧵 **Thread ID**: `{st.session_state.thread_id}`")


# Build agent once
if "agent" not in st.session_state:
    st.session_state.agent = build_graph(user_id=user_id, store=store, retriever=retriever, llm=llm, embeddings=embeddings)

agent = st.session_state.agent


def render_assistant_output(agent_result, entry_index=0):
    for m in agent_result["messages"]:
        if hasattr(m, "type") and m.type in {"ai", "assistant"} and hasattr(m, "content") and m.content:
            st.markdown(m.content)
            return
    # fallback
    st.markdown("_No assistant response generated._")



# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_user_prompt" not in st.session_state:
    st.session_state.pending_user_prompt = None


def main():
    st.markdown(
    """
    <style>
    /* Zoom everything by increasing the base font size */
    html, body, [class*="css"] {
        font-size: 22px !important;
    }

    /* Expand the app content width */
    .main .block-container {
        max-width: 95% !important;
        padding-left: 2rem;
        padding-right: 2rem;
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

    st.title("Insurance Intelligence Assistant")
    st.markdown("Ask your claims, policy, or litigation-related question below:")

    # --- Input Box ---
    user_prompt = st.chat_input("Ask your query...")
    if user_prompt:
        st.session_state.pending_user_prompt = user_prompt
        st.rerun()

    # --- Render Chat History ---
    for idx, entry in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(entry["user_query"])
        with st.chat_message("assistant"):
            render_assistant_output(entry["agent_result"], entry_index=idx)

    # --- New Prompt Execution ---
    if st.session_state.get("pending_user_prompt"):
        prompt = st.session_state.pending_user_prompt
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                config = {
                    "configurable": {
                        "user_id": st.session_state.user_id,
                        "thread_id": st.session_state.thread_id,
                    }
                }
                messages = [{"type": "human", "content": prompt}]
                agent_result = agent.invoke({"messages": messages}, config=config)

            render_assistant_output(agent_result)

        # Save to chat history
        st.session_state.chat_history.append({
            "user_query": prompt,
            "agent_result": agent_result
        })
        st.session_state.pending_user_prompt = None
        st.rerun()


if __name__ == "__main__":
    main()