import streamlit as st
import os, re, uuid, time, itertools
from typing import List, TypedDict
from google import genai
from google.genai import types # Required for safety settings
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Missing API Key.")
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

# --- 3. SAFETY CONFIG ---
# This relaxes filters to prevent "ClientError" on technical/modding discussions
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
        "SYSTEM: You are a Technical Auditor. Analyze files (DLL, Code, Audio) for technical patterns.",
        *state['media_handles'],
        f"USER QUERY: {state['question']} {revision}"
    ]
    
    try:
        response = client.models.generate_content(
            model=state['gen_model'], 
            contents=content_parts,
            config=types.GenerateContentConfig(
                safety_settings=safety_config,
                temperature=0.7
            )
        )
        return {"answer": response.text, "attempts": state['attempts'] + 1}
    except Exception as e:
        # If blocked by safety, provide a clean fallback instead of crashing
        return {"answer": f"⚠️ Generator failed or was blocked: {str(e)}", "attempts": state['attempts'] + 1}

def judge_node(state: AgentState):
    eval_prompt = (
        f"Critically audit the response below. Is it technically accurate?\n"
        f"Return SCORE: [0-10000] and CRITIQUE: [text].\n\n"
        f"QUERY: {state['question']}\n"
        f"RESPONSE: {state['answer']}"
    )
    
    try:
        # Judge node doesn't need media_handles; it just audits the text
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
        return {"score": 0, "feedback": f"Judge error: {str(e)}", "attempts": state['attempts']}

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
if "online_models" not in st.session_state: st.session_state.online_models = []

with st.sidebar:
    st.header("⚙️ 1. Model Configuration")
    if st.button("🔍 Check Model Connectivity"):
        with st.status("Probing API...") as status:
            online = []
            for m in ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemma-3-27b-it", "gemma-3-4b-it"]: # Test key stable ones
                try:
                    client.models.generate_content(model=m, contents="hi", config={'max_output_tokens': 1})
                    online.append(m)
                except: continue
            st.session_state.online_models = online
            status.update(label="Complete", state="complete")

    if st.session_state.online_models:
        sel_gen = st.selectbox("Chatting/Parsing Model", st.session_state.online_models)
        sel_judge = st.selectbox("Judge/Auditing Model", st.session_state.online_models, index=len(st.session_state.online_models)-1)
    else:
        sel_gen, sel_judge = None, None

    st.divider()
    st.header("📁 2. Upload Context")
    uploaded_files = st.file_uploader("Upload assets", accept_multiple_files=True)

# --- CHAT INTERFACE ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if query := st.chat_input("Start Technical Audit..."):
    if not (sel_gen and sel_judge):
        st.error("Select models first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)

        with st.chat_message("assistant"):
            with st.status("🚀 Agentic Loop running...") as status:
                # Process files
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
                try:
                    for output in app_compiled.stream(initial_state):
                        for node, data in output.items():
                            if node == "generator":
                                st.write(f"📝 **{sel_gen}** is drafting...")
                                final_ans = data.get('answer', "")
                            elif node == "judge":
                                st.write(f"⚖️ **{sel_judge}** score: **{data.get('score', 0)}**")
                except Exception as loop_error:
                    final_ans = f"Workflow Error: {str(loop_error)}"

                status.update(label="✅ Finished", state="complete")
            
            st.markdown(final_ans)
            st.session_state.chat_history.append({"role": "assistant", "content": final_ans})
