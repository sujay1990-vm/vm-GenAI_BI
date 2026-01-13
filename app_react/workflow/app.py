import streamlit as st
import sys
from graph import build_graph, tool_usage_prompt
import streamlit as st
import os
import warnings
from datetime import datetime
import uuid
import time
from io import BytesIO
import base64
from llm import get_llm, get_embedding_model
from rag_worker import retriever
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage
import random
from pathlib import Path
from PIL import Image


llm = get_llm()
embeddings = get_embedding_model()

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="Claims App")

# Size + spacing knobs
BOX_W = 250      # max logo width in px (inside the white box)
BOX_H = 150       # max logo height in px
PADDING = 20      # white margin on all sides in px
RADIUS = 10      # corner rounding

# --- Robust logo finder ---
def _find_logo():
    here = Path(__file__).parent
    candidates = [
        here / "logo.png",
        here / "logo.PNG",
        here / "assets" / "logo.png",
        here / "assets" / "logo.PNG",
        Path("/dbfs/FileStore/logo.png"),
        Path("/dbfs/FileStore/logo.PNG"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return Image.open(p), str(p)
            except Exception:
                pass
    return None, None

_logo_img, _logo_path = _find_logo()

def _pil_to_data_uri(img):
    if img is None:
        return None
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

left, right = st.columns([2, 14], gap="large")

# with left:
#     if _logo_img:
#         data_uri = _pil_to_data_uri(_logo_img)
#         st.markdown(
#             f"""
#             <div style="
#                 background: #fff;
#                 padding: {PADDING}px;
#                 border-radius: {RADIUS}px;
#                 display: inline-flex;
#                 align-items: center;
#                 justify-content: center;
#             ">
#                 <img src="{data_uri}" alt="logo"
#                      style="display:block; max-width:{BOX_W}px; max-height:{BOX_H}px; object-fit:contain;"/>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     else:
#         st.write("🧱")

# with right:
#     st.markdown(
#         "<h1 style='margin:0; line-height:80px;'>Claims knowledge management solution</h1>",
#         unsafe_allow_html=True
#     )

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
with st.container():
    data_uri = _pil_to_data_uri(_logo_img) if _logo_img else None
    st.markdown(
        f"""
        <div class="header-flex">
            <div class="logo-box">
                {"<img src='" + data_uri + "' alt='logo'>" if data_uri else "🧱"}
            </div>
            <div class="header-title">
                <h1>Claims knowledge management solution</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
# st.title("Claims knowledge management solution")

# 🔐 Session Setup: User ID & Thread ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id  # ✅ Always set this outside the if block



if "memory_store" not in st.session_state:
    st.session_state.memory_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["user_query", "final_response"]
    }
        )

store = st.session_state.memory_store



def generate_thread_id():
    return str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

## Reset Thread

with st.container():
    st.markdown('<div class="clear-button-container">', unsafe_allow_html=True)

    if st.button("🧹 Clear History"):
        st.session_state.thread_id = generate_thread_id()
        st.session_state.chat_history = []
        st.session_state.prompt_count = 0
        st.toast("✅ Started a new thread!")

    st.markdown('</div>', unsafe_allow_html=True)



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


    # Step 4: Render final output
    if final_output:
        st.markdown(final_output)
    else:
        st.markdown("_No assistant response generated._")

    # ✅ Render all SystemMessages for follow-up and confidence
    confidence_score_msg = None
    confidence_reasoning_msg = None

    # Step 3: Render tool calls (if any)
    if tool_traces:
        joined_traces = "\n\n".join(tool_traces)
        with st.expander("🧠 Reasoning (Tool Calls)", expanded=False):
            st.markdown(f"```text\n{joined_traces}\n```")

    # Find latest follow-up message
    latest_followup_msg = next(
        (m for m in reversed(messages)
        if m.type == "system" and "follow-up" in m.content.lower()),
        None
    )

    # Find latest confidence and reasoning messages
    for m in reversed(messages):
        if m.type == "system":
            content_lower = m.content.lower()

            if "final answer confidence" in content_lower and confidence_score_msg is None:
                confidence_score_msg = m.content

            elif "explanation" in content_lower and confidence_reasoning_msg is None:
                confidence_reasoning_msg = m.content

    # ✅ Render follow-up
    if latest_followup_msg:
        with st.expander("💡 Suggested Follow-Up Questions", expanded=False):
            st.markdown(latest_followup_msg.content)

    # ✅ Render confidence + reasoning
    if confidence_score_msg or confidence_reasoning_msg:
        with st.expander("✅ Confidence in Final Answer", expanded=False):
            if confidence_score_msg:
                st.markdown(confidence_score_msg)
            if confidence_reasoning_msg:
                st.markdown(confidence_reasoning_msg)


# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_user_prompt" not in st.session_state:
    st.session_state.pending_user_prompt = None

# ✅ Add this:
if "prompt_count" not in st.session_state:
    st.session_state.prompt_count = 0


if "reset_next" not in st.session_state:
    st.session_state.reset_next = False

def main():
    st.markdown("""
        <style>
        div.stButton > button {
            width: 100%;
            padding: 0.35rem 0.5rem;
            white-space: normal;     /* allow wrapping inside button */
            line-height: 1.15;
        }       
        .header-flex {
            display: flex;
            align-items: center;
            gap: 24px;  /* controls space between logo and title */
            flex-wrap: wrap; /* allows wrapping on small screens / zoom-in */
        }
        .logo-box {
            background: #fff;
            padding: 20px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            max-width: 250px;
            max-height: 150px;
        }
        .logo-box img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: block;
        }
        .header-title h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.3;
        }
        .clear-button-container {
            margin-top: 20px;  /* 🔼 increase this value for more spacing */
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("Ask your claims, policy, or guidelines related question below:")
    # st.markdown(f"🧠 **Current Thread ID**: `{st.session_state.thread_id}`")
    # st.markdown(f"🧠 **Current User ID**: `{st.session_state.user_id}`")
        # --- Sample Questions ---
    st.markdown('<h3 style="font-size:30px; font-weight:700;">💬 Sample Questions</h3>', unsafe_allow_html=True)
    all_questions = [
        # "Can you summarize entire claim details for Vicki Morgan",
        # "what is the threshold value for Total loss and What is the Soft threshold damage amount for vehicles?",
        # "What is the litigation risk for claim 81AAD85F",
        # "Does claim number C9CB6205 classify as a complex BI claim?",
        # "What actions or strategies can be taken to maximize recovery for claims with high subrogation propensity scores like 0.8?",
        # "What documents are available for claim 587DED91 ?",
        # "Can you show avg recovery rate by states ?",
        # "what is subrogation propensity score for claim 587DED91 ?",
        # "What are claims that are similar to 587DED91 ?" ,
        # "Give recovery stats for claim 24FB7D64 ?"  
        "What is the subrogation potential (propensity score) for Claim #587DED91", 
        "What facts support pursuing subrogation ?",
        "What documents do we have and what is missing for this demand ?",
        "Can you show some examples of claims that look like Claim #587DED91.", 
        "The at-fault driver’s carrier is Nelson Group, what is our average recovery rate with them?"
    ]

    if "sample_questions" not in st.session_state:
        st.session_state.sample_questions = random.sample(all_questions, 5)

        # ---- layout controls ----
    PER_ROW = 3  # change to 2/4/etc

    qs = st.session_state.sample_questions
    for start in range(0, len(qs), PER_ROW):
        row = qs[start:start + PER_ROW]
        cols = st.columns(len(row))
        for col, q in zip(cols, row):
            with col:
                if st.button(q, key=f"qbtn_{start}_{q}"):
                    st.session_state.pending_user_prompt = q
                    st.rerun()

    # for i, q in enumerate(st.session_state.sample_questions):
    #     if st.button(q, key=f"qbtn_{i}"):
    #         st.session_state.pending_user_prompt = q
    #         st.rerun()

    # if st.button("🔄 Refresh Sample Questions", key="refresh_qs"):
    #     st.session_state.sample_questions = random.sample(all_questions, 5)
    #     st.rerun()

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
                        },
                        "max_tokens": 3000  # 👈 Limit final assistant output
                    }
                    
                messages = []

                for entry in st.session_state.chat_history:
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

        # ✅ Increment count and set reset flag (deferred to next run)
        st.session_state.prompt_count += 1
        if st.session_state.prompt_count >= 3:
            st.toast("♻️ Auto-resetting thread on next question", icon="♻️")
            st.session_state.reset_next = True

        

        # ✅ Optional: Reset thread ID at end of run, but keep chat on screen
        if st.session_state.get("reset_next"):
            st.session_state.thread_id = generate_thread_id()
            st.session_state.prompt_count = 0
            st.session_state.reset_next = False
            print("Resetting thread ID")
            st.toast("✅ New backend thread started", icon="🧠")

        st.session_state.pending_user_prompt = None
        st.rerun()

if __name__ == "__main__":
    main()