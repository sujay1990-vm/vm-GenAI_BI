import streamlit as st
import sys
from graph import build_graph, tool_usage_prompt
import streamlit as st
import os
import warnings
from datetime import datetime
import uuid
import time
from llm import get_llm, get_embedding_model
from rag_worker import retriever
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(layout="wide")

llm = get_llm()
embeddings = get_embedding_model()

warnings.filterwarnings("ignore", category=FutureWarning)

# 🔐 Session Setup: User ID & Thread ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id  # ✅ Always set this outside the if block



if "memory_store" not in st.session_state:
    st.session_state.memory_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["user_query", "reformulated_query", "final_response"]
    }
        )

store = st.session_state.memory_store


def generate_thread_id():
    return str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()


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


# def render_assistant_output(agent_result, entry_index=0):
#     # Safely extract last relevant assistant message
#     assistant_messages = [
#         m for m in agent_result["messages"]
#         if hasattr(m, "type") and m.type in {"ai", "assistant"} and hasattr(m, "content") and m.content
#     ]

#     if assistant_messages:
#         last_msg = assistant_messages[-1]  # 👈 pick the final message
#         st.markdown(last_msg.content)
#     else:
#         st.markdown("_No assistant response generated._")



def render_assistant_output(agent_result, entry_index=0):
    messages = agent_result["messages"]

    # Step 1: Find the last user query (HumanMessage)
    last_user_index = max(
        (i for i, m in enumerate(messages)
         if hasattr(m, "type") and m.type == "human" and hasattr(m, "content")),
        default=-1
    )

    # Step 2: Collect tool calls *after* last user message and *before* final assistant message
    tool_traces = []
    final_output = None
    for i in range(last_user_index + 1, len(messages)):
        m = messages[i]

        if isinstance(m, AIMessage):
            # Tool calls
            tool_calls = m.additional_kwargs.get("tool_calls", [])
            for call in tool_calls:
                tool_name = call["function"]["name"]
                args = call["function"]["arguments"]
                tool_traces.append(f"🛠️ Tool: {tool_name}\n📥 Args: {args}")

            # Also capture the final assistant response
            if hasattr(m, "content") and m.content:
                final_output = m.content.strip()

    # Step 3: Render tool calls (if any)
    if tool_traces:
        joined_traces = "\n\n".join(tool_traces)
        with st.expander("🧠 Agent Reasoning (Tool Calls)", expanded=False):
            st.markdown(f"```text\n{joined_traces}\n```")


    # Step 4: Render final output
    if final_output:
        st.markdown(final_output)
    else:
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
    
    st.title("Claims knowledge management solution")
    st.markdown("Ask your claims, policy, or guidelines related question below:")

    # --- Input Box ---
    # --- Row just above chat input ---
    col1, col2 = st.columns([5, 1])

    with col1:
        st.markdown("#### ")

    with col2:
        if st.button("🧹", help="Clear History and Start New Thread"):
            st.session_state.thread_id = generate_thread_id()
            st.session_state.chat_history = []
            st.success("✅ Started a new thread!")

    # --- Actual Chat Input (stays pinned at bottom) ---
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
                        },
                        "max_tokens": 3000  # 👈 Limit final assistant output
                    }
                    
                messages = []

                # Keep only last 5 exchanges (10 messages total)
                recent_history = st.session_state.chat_history[-5:]

                for entry in recent_history:
                    messages.append(HumanMessage(content=entry["user_query"]))
                    for m in entry["agent_result"]["messages"]:
                        if hasattr(m, "type") and m.type in {"ai", "assistant"} and hasattr(m, "content"):
                            messages.append(AIMessage(content=m.content))

                # Add the latest user message
                messages.append(HumanMessage(content=prompt))


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