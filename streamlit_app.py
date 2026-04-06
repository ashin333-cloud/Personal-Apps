import streamlit as st
import os, re, uuid, time
from typing import List, TypedDict
from google import genai
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
# load_dotenv() handles local .env files; st.secrets handles Streamlit Cloud
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Missing API Key. Please add GOOGLE_API_KEY to your Streamlit Secrets or .env file.")
    st.stop()

# Initialize the Google GenAI Client
client = genai.Client(api_key=API_KEY)

# --- 2. UNIVERSAL STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    answer: str
    media_handles: List[any] 
    score: int
    attempts: int
    history: List[dict]
    feedback: str

# --- 3. NODE LOGIC (SYNCHRONOUS FOR STABILITY) ---

def universal_generator_node(state: AgentState):
    """Generates a technical audit based on assets and previous feedback."""
    revision_context = f"\nPREVIOUS AUDIT FEEDBACK (Fix these issues): {state['feedback']}" if state.get('feedback') else ""
    
    prompt_parts = [
        "SYSTEM: You are a Universal Technical Auditor. Analyze the provided assets and answer the user query with high technical precision.",
        *state['media_handles'],
        f"USER QUERY: {state['question']} {revision_context}"
    ]
    
    # Using Gemini 2.0 Flash
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=prompt_parts
    )
    return {"answer": response.text, "attempts": state['attempts'] + 1}

def judge_node(state: AgentState):
    """Evaluates the generator's output and provides a score."""
    eval_prompt = (
        f"Critically audit the following response for technical accuracy. "
        f"Return a score between 0 and 10000. "
        f"Format: SCORE: [number] CRITIQUE: [text]\n\n"
        f"RESPONSE TO AUDIT: {state['answer']}"
    )
    
    # Using Gemini 1.5 Flash for the Judge to save costs/latency
    response = client.models.generate_content(
        model="gemini-1.5-flash", 
        contents=[*state['media_handles'], eval_prompt]
    )
    
    # Extract Score and Feedback using Regex
    score_match = re.search(r'SCORE:\s*(\d+)', response.text)
    score = int(score_match.group(1)) if score_match else 0
    
    critique_match = re.search(r'CRITIQUE:\s*(.*)', response.text, re.DOTALL)
    critique = critique_match.group(1).strip() if critique_match else "No specific critique provided."
    
    return {
        "score": score, 
        "feedback": critique,
        "history": state.get('history', []) + [{"attempt": state['attempts'], "score": score, "feedback": critique}]
    }

# --- 4. LANGGRAPH ORCHESTRATION ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("generator", universal_generator_node)
workflow.add_node("judge", judge_node)

# Define Edges
workflow.set_entry_point("generator")
workflow.add_edge("generator", "judge")

# Conditional Logic: Loop back if score < 8500 and attempts < 3
def should_continue(state: AgentState):
    if state['score'] >= 8500 or state['attempts'] >= 3:
        return END
    return "generator"

workflow.add_conditional_edges("judge", should_continue)

# Compile the Graph
app_compiled = workflow.compile()

# --- 5. STREAMLIT PRODUCTION UI ---

st.set_page_config(page_title="Synapse-Native Omni", page_icon="🤖", layout="wide")

st.title("🤖 Synapse-Native: Universal Technical Auditor")
st.markdown("---")

# Sidebar for Asset Uploads
with st.sidebar:
    st.header("1. Upload Context")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, Images, Audio, Video)", 
        accept_multiple_files=True
    )
    st.info("Files are processed securely via Gemini File API.")

# Main Chat Interface
if query := st.chat_input("What would you like me to audit?"):
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one asset in the sidebar to provide context for the audit.")
    else:
        with st.status("🚀 Initializing Agentic Audit Loop...") as status:
            # Step A: Handle Gemini File Uploads
            handles = []
            for f in uploaded_files:
                temp_file = f"temp_{uuid.uuid4()}_{f.name}"
                with open(temp_file, "wb") as buffer:
                    buffer.write(f.getvalue())
                
                # Upload to Gemini
                h = client.files.upload(file=temp_file)
                while h.state.name == "PROCESSING":
                    time.sleep(2)
                    h = client.files.get(name=h.name)
                
                handles.append(h)
                os.remove(temp_file) # Clean up local storage

            # Step B: Execute LangGraph (Synchronous Invoke)
            status.update(label="🧠 Analyzing and Judging (Multi-step loop)...")
            
            initial_state = {
                "question": query, 
                "media_handles": handles, 
                "attempts": 0, 
                "history": [], 
                "score": 0, 
                "feedback": ""
            }
            
            final_output = app_compiled.invoke(initial_state)
            
            status.update(label=f"✅ Audit Complete! Final Score: {final_output['score']}/10000", state="complete")

            # Step C: Display Results
            st.subheader("Final Technical Audit")
            st.markdown(final_output['answer'])
            
            with st.expander("View Agent Reasoning & Iteration Logs"):
                for entry in final_output['history']:
                    st.write(f"**Attempt {entry['attempt']}** | **Score:** {entry['score']}")
                    st.write(f"*Feedback:* {entry['feedback']}")
                    st.divider()
