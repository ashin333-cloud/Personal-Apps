import streamlit as st
import os, re, uuid, time, itertools
from typing import List, TypedDict
from google import genai
from google.genai import types 
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
    gen_model: str 
    judge_model: str 

# --- 3. THE COMPLETE MODEL LIST ---
FULL_MODEL_LIST = [
    "gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001", "gemini-2.0-flash-exp-image-generation",
    "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite",
    "gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts",
    "gemini-2.5-flash-image", "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools", "gemini-3.1-flash-lite-preview",
    "gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it",
    "gemma-3n-e2b-it", "gemma-3n-e4b-it", "gemini-flash-latest",
    "gemini-pro-latest"
]

safety_config = [
    types.SafetySetting(category="HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

# --- 4. NODE LOGIC ---

def universal_generator_node(state: AgentState):
    revision = f"\nPREVIOUS FEEDBACK: {state['feedback']}" if state.get('feedback') else ""
    content_parts = [
        "SYSTEM: You are a Technical Auditor. Analyze the files strictly.",
        *state['media_handles'],
        f"USER QUERY: {state['question']} {revision}"
    ]
    
    try:
        response = client.models.generate_content(
            model=state['gen_model'], 
            contents=content_parts,
            config=types.GenerateContentConfig(safety_settings=safety_config)
        )
        return {"answer": response.text, "attempts": state['attempts'] + 1}
    except Exception as e:
        return {"answer": f"⚠️ Generator Error: {str(e)}", "attempts": state['attempts'] + 1}

def judge_node(state: AgentState):
    eval_prompt = (
        f"Critically audit this response. Return SCORE: [0-10000] and CRITIQUE: [text].\n\n"
        f"USER QUERY: {state['question']}\n"
        f"RESPONSE TO AUDIT: {state['answer']}"
    )
    
    try:
        response = client.models.generate_content(
            model=state['judge_model'], 
            contents=[eval_prompt],
            config=types.GenerateContentConfig(safety_settings=safety_config)
        )
        score_match = re.search(r'SCORE:\s*(\d+)', response.text)
        score = int(score_match.group(1)) if score_match else 0
        critique = re.search(r'CRITIQUE:\s*(.*)', response.text, re.DOTALL).group(1).strip() if "CRITIQUE:" in response.text else "N/A"
        
        return {
            "score": score, "feedback": critique,
            "history": state.get('history', []) + [{"attempt": state['attempts'], "score": score, "feedback": critique}]
        }
    except Exception as e:
        return {"score": 0, "feedback": f"Judge Error: {str(e)}", "attempts": state['attempts']}

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

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_status" not in st.session_state: st.session_state.last_status = {}

with st.sidebar:
    st.header("⚙️ 1. Model Configuration")
    
    # Optional Diagnostic Button
    if st.button("🔍 Diagnostic: Test Connectivity"):
        with st.status("Testing all 24 models...") as status:
            results = {}
            for m in FULL_MODEL_LIST:
                try:
                    client.models.generate_content(model=m, contents="hi", config={'max_output_tokens': 1})
                    results[m] = "🟢"
                except:
                    results[m] = "🔴"
            st.session_state.last_status = results
            status.update(label="Diagnostic Complete", state="complete")

    # Display Diagnostic Results if they exist
    if st.session_state.last_status:
        with st.expander("View Connectivity Log"):
            for m, status in st.session_state.last_status.items():
                st.write(f"{status} {m}")

    # Dropdowns now pull from FULL_MODEL_LIST by default (Not a constraint anymore)
    sel_gen = st.selectbox("Chatting/Parsing Model", FULL_MODEL_LIST, index=5) # Default: 2.5-flash
    sel_judge = st.selectbox("Judge/Auditing Model", FULL_MODEL_LIST, index=20) # Default: gemma-3-27b-it

    st.divider()
    st.header("📁 2. Upload Context")
    uploaded_files = st.file_uploader("Upload assets", accept_multiple_files=True)
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# --- MAIN CHAT ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if query := st.chat_input("Start Technical Audit..."):
    if not uploaded_files:
        st.error("Please upload assets for context.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)

        with st.chat_message("assistant"):
            with st.status("🚀 Agentic Loop running...") as status:
                handles = []
                for f in uploaded_files:
                    t_path = f"tmp_{uuid.uuid4()}_{f.name}"
                    with open(t_path, "wb") as b: b.write(f.getvalue())
                    h = client.files.upload(file=t_path)
                    while h.state.name == "PROCESSING": time.sleep(1); h = client.files.get(name=h.name)
                    handles.append(h)
                    os.remove(t_path)

                initial_state = {
                    "question": query, "media_handles": handles, "attempts": 0, 
                    "history": [], "score": 0, "feedback": "",
                    "gen_model": sel_gen, "judge_model": sel_judge
                }

                final_ans = ""
                for output in app_compiled.stream(initial_state):
                    for node, data in output.items():
                        if node == "generator":
                            st.write(f"📝 **{sel_gen}** analysis step...")
                            final_ans = data.get('answer', "")
                        elif node == "judge":
                            st.write(f"⚖️ **{sel_judge}** score: **{data.get('score', 0)}**")

                status.update(label="✅ Audit Finalized", state="complete")
            
            st.markdown(final_ans)
            st.session_state.chat_history.append({"role": "assistant", "content": final_ans})
