import streamlit as st
import os, re, uuid, time, itertools
from typing import List, TypedDict
from google import genai
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Missing API Key. Check Streamlit Secrets or .env")
    st.stop()

client = genai.Client(api_key=API_KEY)

# --- 2. STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    answer: str
    media_handles: List[any] 
    score: int
    attempts: int
    history: List[dict]
    feedback: str
    gen_model: str # User-selected
    judge_model: str # User-selected

# --- 3. MODEL LIST (VERIFIED 2026) ---
MODELS_TO_TEST = [
    "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", 
    "gemini-2.5-flash-lite", "gemma-3-27b-it", "gemini-3-flash-preview", 
    "gemini-flash-latest", "gemini-pro-latest"
]

# --- 4. NODE LOGIC ---

def universal_generator_node(state: AgentState):
    revision = f"\nPREVIOUS FEEDBACK: {state['feedback']}" if state.get('feedback') else ""
    content_parts = [
        "SYSTEM: You are a Technical Auditor. Analyze the files strictly and answer the query.",
        *state['media_handles'],
        f"USER QUERY: {state['question']} {revision}"
    ]
    
    # Dynamically uses the selected Generator model
    response = client.models.generate_content(
        model=state['gen_model'], 
        contents=content_parts
    )
    return {"answer": response.text, "attempts": state['attempts'] + 1}

def judge_node(state: AgentState):
    eval_prompt = f"Return SCORE: [0-10000] and CRITIQUE: [text]. Audit: {state['answer']}"
    
    # Dynamically uses the selected Judge model
    response = client.models.generate_content(
        model=state['judge_model'], 
        contents=[*state['media_handles'], eval_prompt]
    )
    
    score_match = re.search(r'SCORE:\s*(\d+)', response.text)
    score = int(score_match.group(1)) if score_match else 0
    critique = re.search(r'CRITIQUE:\s*(.*)', response.text, re.DOTALL).group(1).strip() if "CRITIQUE:" in response.text else "N/A"
    
    return {
        "score": score, "feedback": critique,
        "history": state.get('history', []) + [{"attempt": state['attempts'], "score": score, "feedback": critique}]
    }

# --- 5. ORCHESTRATION ---
workflow = StateGraph(AgentState)
workflow.add_node("generator", universal_generator_node)
workflow.add_node("judge", judge_node)
workflow.set_entry_point("generator")
workflow.add_edge("generator", "judge")
workflow.add_conditional_edges("judge", lambda x: END if x['score'] >= 9000 or x['attempts'] >= 3 else "generator")
app_compiled = workflow.compile()

# --- 6. STREAMLIT UI ---
st.set_page_config(page_title="Synapse-Native Omni", layout="wide")
st.title("🤖 Synapse-Native: Universal Technical Auditor")

# Persistence
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "online_models" not in st.session_state: st.session_state.online_models = []

with st.sidebar:
    st.header("⚙️ 1. Model Configuration")
    
    if st.button("🔍 Check Model Connectivity"):
        with st.status("Probing Google Cloud API...") as status:
            online = []
            for m in MODELS_TO_TEST:
                try:
                    client.models.generate_content(model=m, contents="hi", config={'max_output_tokens': 1})
                    online.append(m)
                except: continue
            st.session_state.online_models = online
            status.update(label=f"Done! {len(online)} Models Online", state="complete")

    # Dynamic Choice: Only shows models that passed the check
    if st.session_state.online_models:
        sel_gen = st.selectbox("Chatting/Parsing Model", st.session_state.online_models, index=0)
        sel_judge = st.selectbox("Judge/Auditing Model", st.session_state.online_models, index=min(1, len(st.session_state.online_models)-1))
    else:
        st.warning("Run connectivity check to select models.")
        sel_gen, sel_judge = None, None

    st.divider()
    st.header("📁 2. Upload Context")
    uploaded_files = st.file_uploader("Upload PDF, Code, Images", accept_multiple_files=True)
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if query := st.chat_input("Start Technical Audit..."):
    if not sel_gen or not sel_judge:
        st.error("Please select models in the sidebar before running.")
    elif not uploaded_files:
        st.error("Please upload assets for context.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)

        with st.chat_message("assistant"):
            with st.status("🚀 Agentic Loop running...") as status:
                # File Handling
                handles = []
                for f in uploaded_files:
                    t_path = f"tmp_{uuid.uuid4()}_{f.name}"
                    with open(t_path, "wb") as b: b.write(f.getvalue())
                    h = client.files.upload(file=t_path)
                    while h.state.name == "PROCESSING": time.sleep(1); h = client.files.get(name=h.name)
                    handles.append(h)
                    os.remove(t_path)

                # Execute with chosen models
                initial_state = {
                    "question": query, "media_handles": handles, "attempts": 0, 
                    "history": [], "score": 0, "feedback": "",
                    "gen_model": sel_gen, "judge_model": sel_judge
                }

                final_ans = ""
                for output in app_compiled.stream(initial_state):
                    for node, data in output.items():
                        if node == "generator":
                            st.write(f"📝 **{sel_gen}** is drafting analysis...")
                            final_ans = data['answer']
                        elif node == "judge":
                            st.write(f"⚖️ **{sel_judge}** scored this: **{data['score']}**")

                status.update(label="✅ Audit Finalized", state="complete")
            
            st.markdown(final_ans)
            st.session_state.chat_history.append({"role": "assistant", "content": final_ans})
