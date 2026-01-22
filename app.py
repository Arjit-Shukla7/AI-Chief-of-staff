import os
import datetime
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from langchain_google_community import GmailToolkit, CalendarToolkit
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver 
from langchain.tools import tool
from langchain_core.tools import StructuredTool

load_dotenv()

def build_resource_service(credentials):
    """Builds the Calendar API service."""
    return build("calendar", "v3", credentials=credentials)

def build_gmail_service(credentials):
    """Builds the Gmail API service."""
    return build("gmail", "v1", credentials=credentials)

class AssistantBackend:
    def __init__(self, token_file="token.json"):
        self.scopes = [
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/gmail.modify"
        ]
        self.creds = self._load_credentials(token_file)
        self.llm = self._setup_llm()
        self.checkpointer = InMemorySaver()
        
        self.calendar_service = build_resource_service(credentials=self.creds)
        self.calendar_toolkit = CalendarToolkit(api_resource=self.calendar_service)
        self.calendar_tools = self.calendar_toolkit.get_tools()
        
        self.gmail_service = build_gmail_service(credentials=self.creds)
        self.gmail_toolkit = GmailToolkit(api_resource=self.gmail_service)
        self.gmail_tools = self.gmail_toolkit.get_tools()
        
        self.web_search = TavilySearch(k=3)
        
        self.calendar_agent = self._create_calendar_agent()
        self.email_agent = self._create_email_agent()
        self.web_agent = self._create_web_agent()
        
        self.supervisor_agent = self._create_supervisor_agent()

    def _load_credentials(self, token_file):
        if os.path.exists(token_file):
            return Credentials.from_authorized_user_file(token_file, self.scopes)
        else:
            raise FileNotFoundError(f"Token file {token_file} not found. Please authenticate.")

    def _setup_llm(self):
        return ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0,
            reasoning_format="parsed",
        )

    def _create_calendar_agent(self):
        today = datetime.datetime.now().strftime("%A, %B %d, %Y")
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        
        CALENDAR_AGENT_PROMPT = f"""
        You are a calendar scheduling assistant.
        current date:{today}
        current time:{current_time}
        Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') 
        always use the current date and time for finding relationship between date and time,Time is always in IST
        into proper ISO datetime formats.
        you have a wide varity of tools like 'create_calendar_event', 'search_events', 'update_calendar_event', 'get_calendars_info', 'move_calendar_event', 'delete_calendar_event', 'get_current_datetime',
        when asked to create an event never use search_events tool
        """
        return create_agent(
            self.llm,
            tools=self.calendar_tools,
            system_prompt=CALENDAR_AGENT_PROMPT,
        )

    def _create_email_agent(self):
        EMAIL_AGENT_PROMPT = """
        "You are an email assistant. "
        "Compose professional emails based on natural language requests. "
        "Extract recipient information and craft appropriate subject lines and body text. "
        "Use send_email to send the message. "
        "when asked to create a email draft always use create_gmail_draft tool"
        """
        return create_agent(
            self.llm,
            tools=self.gmail_tools, 
            system_prompt=EMAIL_AGENT_PROMPT, 
            middleware=[ 
                HumanInTheLoopMiddleware( 
                    interrupt_on={"send_gmail_message": True}, 
                    description_prefix="mail sending pending approval", 
                ), 
            ],
            checkpointer=self.checkpointer,  
        )

    def _create_web_agent(self):
        WEB_SEARCH_PROMPT="""
            you are an websearch assistant
            you can search web and provide the relatable information the user need
        """
        return create_agent(
            self.llm,
            tools=[self.web_search],
            system_prompt=WEB_SEARCH_PROMPT
        )

    def _schedule_event_tool(self, request: str) -> str:
        """Schedule calendar events using natural language."""
        result = self.calendar_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].text

    def _manage_email_tool(self, request: str) -> str:
        """Send emails using natural language."""
        result = self.email_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].text

    def _web_tool(self, request: str) -> str:
        """Return back the result after searching the internet."""
        result = self.web_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].text

    def _create_supervisor_agent(self):
        today = datetime.datetime.now().strftime("%A, %B %d, %Y")
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        
        SUPERVISOR_PROMPT = f"""
        You are an advanced "Chief of Staff" AI Agent. 
        Current Date: {today}
        Current Time: {current_time}

        Your Goal: Manage the user's life efficiently using Calendar, Email, and Web Search.
        "You can schedule calendar events ,send emails and search web "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "When a request involves multiple actions, use multiple tools in sequence."
        1. **"DAILY BRIEFING"**:
            - Check Calendar: List today's meetings.
            - Check Gmail: Search for unread emails from "important" senders or subjects like "Urgent".
            - Check Web: Search for "Weather in Roorkee,Uttrakhand" and top tech news headlines.
            - Output: A concise 3-section morning report.
        """
        
        tools = [
            StructuredTool.from_function(
                func=self._schedule_event_tool,
                name="schedule_event",
                description="Use this when the user wants to create, modify, or check calendar appointments."
            ),
            StructuredTool.from_function(
                func=self._manage_email_tool,
                name="manage_email",
                description="Use this when the user wants to send notifications, reminders, or any email communication."
            ),
            StructuredTool.from_function(
                func=self._web_tool,
                name="web",
                description="Use this when you want the latest information from the web."
            )
        ]

        return create_agent(
            self.llm,
            tools=tools,
            system_prompt=SUPERVISOR_PROMPT,
            checkpointer=self.checkpointer, 
        )

    def process_query(self, query: str, thread_id: str = "6"):
        """
        Executes the query with Human-in-the-Loop logic.
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"--- Processing Query: {query} ---")
        
        interrupts = []
        for step in self.supervisor_agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config,
        ):
            self._print_step_update(step, interrupts)

        if interrupts:
            resume_payload = self._handle_human_approval(interrupts)
            if resume_payload:
                print("\n--- Resuming Execution ---")
                for step in self.supervisor_agent.stream(
                    Command(resume=resume_payload), 
                    config,
                ):
                    self._print_step_update(step, interrupts=[])

    def _print_step_update(self, step, interrupts):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            else:
                interrupt_ = update[0]
                interrupts.append(interrupt_)
                print(f"\nINTERRUPTED: {interrupt_.id}")

    def _handle_human_approval(self, interrupts):
        """
        Iterates through interrupts and asks user for approval or edits.
        """
        resume = {}
        for interrupt_ in interrupts:
            action_requests = interrupt_.value.get("action_requests", [])
            
            for request in action_requests:
                print(f"\nPENDING ACTION ({interrupt_.id}):")
                print(f"Action: {request.get('description', 'Unknown Action')}")
                
                current_args = request.get("args") or request.get("tool_input") or request.get("arguments")
                
                if current_args is None and "action" in request:
                    action_obj = request["action"]
                    if hasattr(action_obj, "tool_input"):
                        current_args = action_obj.tool_input
                
                if current_args is None:
                    current_args = {} 
                    
                print(f"Arguments: {current_args}")
                
                decision = input("Do you want to Approve (a) or Edit (e) this action? [a/e]: ").strip().lower()
                
                if decision == 'e':
                    print("Current Arguments:", current_args)
                    
                    new_subject = input(f"Enter new subject (current: {current_args.get('subject', '')}): ")
                    new_body = input(f"Enter new body (current: {current_args.get('message', '')}): ")
                    
                    edited_request = request.copy()
                    
                    new_args = current_args.copy()
                    if new_subject: new_args['subject'] = new_subject
                    if new_body: new_args['message'] = new_body 
                    
                    if "args" in edited_request:
                        edited_request["args"] = new_args
                    elif "tool_input" in edited_request:
                        edited_request["tool_input"] = new_args
                    elif "arguments" in edited_request:
                        edited_request["arguments"] = new_args
                    elif "action" in edited_request:                                                      
                        pass

                    edited_request["args"] = new_args 

                    resume[interrupt_.id] = {
                        "decisions": [{"type": "edit", "edited_action": edited_request}]
                    }
                    print("Action edited.")
                else:
                    resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}
                    print("Action approved.")
        return resume

if __name__ == "__main__":
    try:
        backend = AssistantBackend()
        user_query = input("Enter your request: ")
        if user_query:
            backend.process_query(user_query)
    except Exception as e:
        print(f"An error occurred: {e}")