import streamlit as st
from app import AssistantBackend
from langgraph.types import Command
import uuid

st.set_page_config(
    page_title="AI Chief of Staff",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    .stChatMessage {
        background-color: transparent;
    }
    .stStatus {
        font-size: 0.8em;
    }
    div[data-testid="stForm"] {
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    div[data-testid="stForm"] {
        background-color: #0f172a; 
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 20px;
    }
    div[data-testid="stForm"] textarea,
    div[data-testid="stForm"] input {
        background-color: #020617; 
        color: #e5e7eb;
        border: 1px solid #1f2937;
        border-radius: 8px;
    }
    div[data-testid="stForm"] button[kind="primary"] {
        background-color: #22c55e; 
        color: black;
        border: none;
        font-weight: 600;
    }
    div[data-testid="stForm"] button:not([kind="primary"]) {
        background-color: #111827;
        color: #e5e7eb;
        border: 1px solid #374151;
    }
    div[data-testid="stForm"] button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_backend():
    """Initialize the backend once."""
    return AssistantBackend()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

backend = get_backend()
config = {"configurable": {"thread_id": st.session_state.thread_id}}


def get_safe_args(request):
    """
    Safely extracts arguments from a tool request, handling different key names
    (args, arguments, tool_input).
    """
    args = request.get("args") or request.get("tool_input") or request.get("arguments")
    

    if args is None and "action" in request:
        if hasattr(request["action"], "tool_input"):
            args = request["action"].tool_input
            
    return args if args is not None else {}

def process_stream(stream_generator):
    """
    Iterates through the agent's stream, updating the UI with thought process
    and handling interrupts.
    """
    with st.chat_message("assistant"):
        with st.status("Processing request...", expanded=True) as status:
            response_placeholder = st.empty()
            final_response = ""
            
            for step in stream_generator:
                for key, update in step.items():
                    if "messages" in update:
                        for msg in update["messages"]:
                            if msg.type == "ai" and msg.tool_calls:
                                for tool in msg.tool_calls:
                                    status.write(f"🛠️ **Executing:** `{tool['name']}`")
                            
                            elif msg.type == "tool":
                                status.write(f"✅ **Tool Output:** {msg.content[:100]}...")

                            elif msg.type == "ai" and msg.content:
                                final_response = msg.content

                    if key == "__interrupt__":
                        interrupt_value = update[0]
                        st.session_state.pending_interrupt = interrupt_value
                        status.update(label="✋ Approval Required", state="error")
                        st.rerun() 

            status.update(label="Complete", state="complete", expanded=False)
            if final_response:
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
with st.sidebar:
    st.header("Controls")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_interrupt = None
        st.rerun()
    st.divider()
    st.markdown("**Active Capabilities:**")
    st.markdown("Google Calendar")
    st.markdown("Gmail (Draft & Send)")
    st.markdown("Web Search (Tavily)")

st.title("AI Chief of Staff")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.pending_interrupt:
    interrupt = st.session_state.pending_interrupt
    action_request = interrupt.value["action_requests"][0]
    func_name = action_request.get("description", "Unknown Action")
    
    args = get_safe_args(action_request) 
    
    with st.container():
        st.info(f"✋ **Action Required:** The agent wants to perform the following action:")
        with st.form(key="approval_form"):
            st.subheader(f"Review: {func_name}")
            
            new_args = args.copy()
            
            if "subject" in args:
                new_args["subject"] = st.text_input("Subject", value=args["subject"])
            if "message" in args: 
                new_args["message"] = st.text_area("Email Body", value=args["message"], height=150)
            elif "body" in args: 
                new_args["body"] = st.text_area("Email Body", value=args["body"], height=150)
                
            if "to" in args:
                st.text(f"To: {args['to']}") 

            col1, col2 = st.columns([1, 1])
            with col1:
                approve_btn = st.form_submit_button("Approve & Continue", type="primary")
            with col2:
                cancel_btn = st.form_submit_button("Reject / Cancel")

        if approve_btn:
            if new_args != args:
                edited_action = action_request.copy()
                if "args" in edited_action:
                    edited_action["args"] = new_args
                elif "tool_input" in edited_action:
                     edited_action["tool_input"] = new_args
                else:
                    edited_action["args"] = new_args
                    
                resume_payload = {
                    interrupt.id: {
                        "decisions": [{"type": "edit", "edited_action": edited_action}]
                    }
                }
            else:
                resume_payload = {
                    interrupt.id: {"decisions": [{"type": "approve"}]}
                }
            st.session_state.pending_interrupt = None
            
            stream = backend.supervisor_agent.stream(
                Command(resume=resume_payload),
                config
            )
            process_stream(stream)
            st.rerun()

        if cancel_btn:
            st.session_state.pending_interrupt = None
            st.error("Action cancelled by user.")
            st.session_state.messages.append({"role": "assistant", "content": "Action cancelled by user."})
            st.rerun()

elif prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    stream = backend.supervisor_agent.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config
    )
    process_stream(stream)