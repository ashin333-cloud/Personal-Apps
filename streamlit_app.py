import streamlit as st
import os, re, asyncio, uuid
from typing import List, TypedDict
from google import genai
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
load_dotenv()
# Logic: Use .env key if available, otherwise use Streamlit Cloud Secrets
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Missing API Key. Add it to Streamlit Secrets or .env")
    st.stop()

client = genai.Client(api_key=API_KEY)

# --- 2. UNIVERSAL STATE (UNTOUCHED LOGIC) ---
class AgentState(TypedDict):
    question: str
    answer: str
    media_handles: List[any] 
    score: int
    attempts: int
    history: List[dict]
    feedback: str

# --- 3. NODES (UNTOUCHED LOGIC) ---
async def universal_generator_node(state: AgentState):
    revision = f"\nPREVIOUS FEEDBACK: {state['feedback']}" if state.get('feedback') else ""
    prompt_parts = [
        "SYSTEM: You are a Universal Technical Auditor.",
        *state['media_handles'],
        f"USER QUESTION: {state['question']} {revision}"
    ]
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt_parts)
    return {"answer": response.text, "attempts": state['attempts'] + 1}

async def judge_node(state: AgentState):
    eval_prompt = f"Score (0-10000) and Critique. Format: SCORE: [int] CRITIQUE: [str]\nRESPONSE: {state['answer']}"
    response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=[*state['media_handles'], eval_prompt])
    
    score_match = re.search(r'SCORE:\s*(\d+)', response.text)
    score = int(score_match.group(1)) if score_match else 0
    critique = re.search(r'CRITIQUE:\s*(.*)', response.text).group(1) if "CRITIQUE:" in response.text else "No feedback."
    
    return {
        "score": score, "feedback": critique,
        "history": state.get('history', []) + [{"score": score, "answer": state['answer']}]
    }

# --- 4. GRAPH CONSTRUCTION (UNTOUCHED LOGIC) ---
workflow = StateGraph(AgentState)
workflow.add_node("generator", universal_generator_node)
workflow.add_node("judge", judge_node)
workflow.set_entry_point("generator")
workflow.add_edge("generator", "judge")
workflow.add_conditional_edges("judge", lambda x: END if x['score'] >= 8500 or x['attempts'] >= 3 else "generator")
app_compiled = workflow.compile()

# --- 5. PRODUCTION UI ---
st.set_page_config(page_title="Synapse-Native Omni", layout="wide")
st.title("🤖 Synapse-Native: Universal Technical Auditor")

uploaded_files = st.sidebar.file_uploader("Upload Assets (PDF, JPG, MP3)", accept_multiple_files=True)

if query := st.chat_input("Enter your technical query..."):
    if not uploaded_files:
        st.warning("Please upload files to proceed.")
    else:
        with st.status("🔍 Executing Agentic Audit Loop...") as status:
            handles = []
            for f in uploaded_files:
                temp_name = f"tmp_{uuid.uuid4()}_{f.name}"
                with open(temp_name, "wb") as b: b.write(f.getvalue())
                h = client.files.upload(file=temp_name)
                while h.state.name == "PROCESSING":
                    asyncio.run(asyncio.sleep(2))
                    h = client.files.get(name=h.name)
                handles.append(h)
                os.remove(temp_name)

            final_state = asyncio.run(app_compiled.ainvoke({
                "question": query, "media_handles": handles, "attempts": 0, 
                "history": [], "score": 0, "feedback": ""
            }))
            
            status.update(label=f"Audit Complete! Final Score: {final_state['score']}", state="complete")
            st.markdown(final_state['answer'])
            with st.expander("Audit Logs"): st.json(final_state['history'])
