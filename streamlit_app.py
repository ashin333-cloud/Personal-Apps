import streamlit as st

import os, re, uuid, time

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



# --- 3. NODE LOGIC ---



def universal_generator_node(state: AgentState):

    # Pulling context from the state and the history to ensure continuity

    revision = f"\nPREVIOUS FEEDBACK: {state['feedback']}" if state.get('feedback') else ""

    

    content_parts = [

        "SYSTEM: You are a Technical Auditor. MANDATORY: Analyze the provided files strictly. "

        "If a 'requirements.txt' is provided, audit it in detail.",

        *state['media_handles'],

        f"USER QUERY: {state['question']} {revision}"

    ]



    response = client.models.generate_content(

        model="gemini-2.5-flash", 

        contents=content_parts

    )

    return {"answer": response.text, "attempts": state['attempts'] + 1}



def judge_node(state: AgentState):

    eval_prompt = (

        f"Critically audit this response. Return SCORE: [0-10000] and CRITIQUE: [text]. "

        f"RESPONSE: {state['answer']}"

    )

    

    response = client.models.generate_content(

        model="gemini-2.5-flash-lite", 

        contents=[*state['media_handles'], eval_prompt]

    )

    

    score = int(re.search(r'SCORE:\s*(\d+)', response.text).group(1)) if re.search(r'SCORE:\s*(\d+)', response.text) else 0

    critique = re.search(r'CRITIQUE:\s*(.*)', response.text, re.DOTALL).group(1).strip() if "CRITIQUE:" in response.text else "N/A"

    

    return {

        "score": score, 

        "feedback": critique,

        "history": state.get('history', []) + [{"attempt": state['attempts'], "score": score, "feedback": critique}]

    }



# --- 4. ORCHESTRATION ---

workflow = StateGraph(AgentState)

workflow.add_node("generator", universal_generator_node)

workflow.add_node("judge", judge_node)

workflow.set_entry_point("generator")

workflow.add_edge("generator", "judge")

workflow.add_conditional_edges("judge", lambda x: END if x['score'] >= 9000 or x['attempts'] >= 3 else "generator")

app_compiled = workflow.compile()



# --- 5. STREAMLIT UI WITH SESSION HISTORY ---



st.set_page_config(page_title="Synapse-Native Omni", layout="wide")

st.title("🤖 Synapse-Native: Agentic Auditor")



# Feature 1: Store Previous Chat History

if "chat_history" not in st.session_state:

    st.session_state.chat_history = [] # Format: {"role": "user/assistant", "content": "text"}



# Sidebar for Assets

with st.sidebar:

    st.header("1. Upload Assets")

    uploaded_files = st.file_uploader("Upload context files", accept_multiple_files=True)

    if st.button("Clear Chat History"):

        st.session_state.chat_history = []

        st.rerun()



# Display existing chat history

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])



# Main Input

if query := st.chat_input("Analyze these files..."):

    # Add user message to history

    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):

        st.markdown(query)



    if not uploaded_files:

        st.error("Please upload assets in the sidebar.")

    else:

        with st.chat_message("assistant"):

            with st.status("🚀 Agentic Audit in Progress...") as status:

                # Step A: File Processing

                handles = []

                for f in uploaded_files:

                    t_file = f"temp_{uuid.uuid4()}_{f.name}"

                    with open(t_file, "wb") as b: b.write(f.getvalue())

                    h = client.files.upload(file=t_file)

                    while h.state.name == "PROCESSING":

                        time.sleep(1)

                        h = client.files.get(name=h.name)

                    handles.append(h)

                    os.remove(t_file)



                # Feature 2: Visible Execution Steps via LangGraph Streaming

                initial_state = {

                    "question": query, "media_handles": handles, "attempts": 0, 

                    "history": [], "score": 0, "feedback": ""

                }



                final_ans = ""

                # We use .stream() to catch each node's output as it happens

                for output in app_compiled.stream(initial_state):

                    for node_name, data in output.items():

                        if node_name == "generator":

                            st.write("📝 **Generator:** Drafted a technical analysis.")

                            final_ans = data['answer']

                        elif node_name == "judge":

                            st.write(f"⚖️ **Judge:** Scored response **{data['score']}/10000**")

                            if data['score'] < 9000:

                                st.write(f"🔄 *Re-routing for improvement:* {data['feedback'][:100]}...")



                status.update(label="✅ Audit Finalized", state="complete")

            

            # Display Final Result

            st.markdown(final_ans)

            st.session_state.chat_history.append({"role": "assistant", "content": final_ans})
