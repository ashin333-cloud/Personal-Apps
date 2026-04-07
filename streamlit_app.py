import streamlit as st
import os, re, uuid, time, itertools
from typing import List, TypedDict
from google import genai
from google.genai import types 
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

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

# --- 3. FULL MODEL LIST ---
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
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

# --- 4. NODE LOGIC ---
def universal_generator_node(state: AgentState):
    revision = f"\nPREVIOUS FEEDBACK: {state['feedback']}" if state.get('feedback') else ""
    content_parts = ["SYSTEM: Technical Auditor.", *state['media_handles'], f"{state['question']} {revision}"]
    try:
        response = client.models.generate_content(
            model=state['gen_model'], contents=content_parts,
            config=types.GenerateContentConfig(safety_settings=safety_config)
        )
        return {"answer": response.text, "attempts": state['attempts'] + 1}
    except Exception as e:
        return {"answer": f"⚠️ Error: {str(e)}", "attempts": state['attempts'] + 1}

def judge_node(state: AgentState):
    eval_prompt = f"SCORE: [0-10000] CRITIQUE: [text]\n\nRESPONSE: {state['answer']}"
    try:
        response = client.models.generate_content(
            model=state['judge_model'], contents=[eval_prompt],
            config=types.GenerateContentConfig(safety_settings=safety_config)
        )
        score_match = re.search(r'SCORE:\s*(\d+)', response.text)
        score = int(score_match.group(1)) if score_match else 0
        return {
            "score": score, "feedback": response.text,
            "history": state.get('history', []) + [{"attempt": state['attempts'], "score": score, "critique": response.text}]
        }
    except Exception as e:
        return {"score": 0, "feedback": f"Judge Error: {str(e)}"}

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
if "test_run_complete" not in st.session_state: st.session_state.test_run_complete = False

with st.sidebar:
    st.header("⚙️ 1. Model Configuration")
    if st.button("🔍 Diagnostic: Test Connectivity"):
        with st.status("Testing Models...") as status:
            online = []
            for i, m in enumerate(FULL_MODEL_LIST):
                try:
                    llm = ChatGoogleGenerativeAI(model=m, google_api_key=API_KEY, timeout=10, max_retries=0)
                    llm.invoke([HumanMessage(content="Hi")])
                    online.append(m)
                    st.write(f"🟢 {m}")
                except: continue
            st.session_state.online_models = online
            st.session_state.test_run_complete = True
            st.rerun()

    display_list = st.session_state.online_models if st.session_state.test_run_complete else FULL_MODEL_LIST
    test_key = "online_v6" if st.session_state.test_run_complete else "full_v6"
    
    if display_list:
        sel_gen = st.selectbox("Chatting/Parsing Model", display_list, index=0, key=f"gen_{test_key}")
        sel_judge = st.selectbox("Judge/Auditing Model", display_list, index=len(display_list)-1, key=f"judge_{test_key}")

    st.divider()
    st.header("📁 2. Upload Context")
    uploaded_files = st.file_uploader("Upload assets", accept_multiple_files=True)

# --- RENDERING ENGINE ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        # Check if there are preserved logs for this specific message
        if "audit_logs" in msg:
            for log in msg["audit_logs"]:
                with st.expander(f"⚖️ Audit Attempt {log['attempt']} | Score: {log['score']}/10000"):
                    st.markdown(log["critique"])
        st.markdown(msg["content"])

# --- PROCESSING ENGINE ---
if query := st.chat_input("Start Technical Audit..."):
    if not uploaded_files:
        st.error("Please upload context files.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)

        with st.chat_message("assistant"):
            current_audit_logs = [] # Temp storage for this specific run
            response_placeholder = st.empty()
            
            with st.status("🚀 Agentic Loop Processing...") as status:
                handles = []
                for f in uploaded_files:
                    t_path = f"tmp_{uuid.uuid4()}_{f.name}"
                    with open(t_path, "wb") as b: b.write(f.getvalue())
                    h = client.files.upload(file=t_path)
                    while h.state.name == "PROCESSING": time.sleep(1); h = client.files.get(name=h.name)
                    handles.append(h)
                    os.remove(t_path)

                final_ans = ""
                for output in app_compiled.stream({
                    "question": query, "media_handles": handles, "attempts": 0, 
                    "gen_model": sel_gen, "judge_model": sel_judge, "feedback": "", "history": []
                }):
                    for node, data in output.items():
                        if node == "generator":
                            final_ans = data.get('answer', "")
                            st.write(f"📝 Attempt {data.get('attempts')} generated.")
                        if node == "judge":
                            # Store the log for persistence
                            log_entry = {
                                "attempt": data.get('attempts', 0),
                                "score": data.get('score', 0),
                                "critique": data.get('feedback', "")
                            }
                            current_audit_logs.append(log_entry)
                            
                            # Immediate UI feedback
                            st.write(f"⚖️ **Score: {log_entry['score']}/10000**")
                
                status.update(label="✅ Audit Finalized", state="complete")
            
            response_placeholder.markdown(final_ans)
            
            # Save the final answer AND the collected logs to session state
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": final_ans,
                "audit_logs": current_audit_logs # THIS PRESERVES THE JUDGE DATA
            })
