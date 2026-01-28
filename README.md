# AI Chief of Staff

An intelligent, agentic workflow automation system that acts as a personal **AI Chief of Staff**. This application orchestrates Google Calendar, Gmail, and Web Search using natural language commands, while incorporating a **Human-in-the-Loop (HITL)** approval mechanism for sensitive actions such as email sending. Built using **LangChain**, **LangGraph**, and **Streamlit**, the project demonstrates real-world agent orchestration, tool delegation, and human oversight in agentic AI systems.

---
## Project launch video(Made in after effects):


https://github.com/user-attachments/assets/f5a5493a-719f-4da4-8d97-e86d2138ba46


## Product Demo:

https://github.com/user-attachments/assets/51b39f35-790e-426c-bac2-aa68fc0f117f


## Key Features

- **Hierarchical Agent Architecture**  
  Supervisor–worker agent pattern implemented with LangGraph to coordinate multiple specialized agents.

- **Human-in-the-Loop (HITL) Approval**  
  Sensitive actions like email sending require explicit user review, editing, and approval before execution.

- **Natural Language Task Execution**  
  Schedule meetings, draft emails, and retrieve web information using conversational English.

- **Interactive Streamlit Interface**  
  Real-time agent streaming, execution status, and approval workflows through a clean UI.

---

## System Architecture

The system follows a multi-agent design where a central Supervisor Agent interprets user intent and delegates tasks to specialized sub-agents:

-  **Calendar Agent** – Manages Google Calendar events  
-  **Email Agent** – Drafts and sends Gmail messages with HITL safeguards  
-  **Web Agent** – Retrieves real-time information using Tavily search  

State is maintained using thread-based conversation tracking with in-memory checkpointing for resumable execution.

---

## Tech Stack

- Python 3.9+
- LangChain
- LangGraph
- Streamlit
- Google Calendar API
- Gmail API
- Groq LLM
- Tavily Search API
- python-dotenv

---

## Project Structure
├── backend.py # Agent logic, tools, workflows, HITL integration
├── streamlit.py # Streamlit UI, streaming responses, approval interface
├── .env # Environment variables (not committed)
├── credentials.json # Google OAuth credentials
├── token.json # Generated OAuth token
└── README.md
## Setup Instructions

### Prerequisites

- Python 3.9 or higher  
- Google Cloud project with Gmail and Calendar APIs enabled  
- OAuth credentials file (`credentials.json`)  
- Groq and Tavily API keys  

### Install Dependencies

```bash
pip install -r requirements.txt
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```
## Example Capabilities

- Schedule meetings using natural language  
- Draft and send emails with human approval  
- Generate daily briefings from calendar, email, and web data  
- Perform real-time web searches  

---

## Human-in-the-Loop Safety

The application enforces explicit human approval to prevent unintended actions, allow manual edits to generated content, and ensure transparency and control over autonomous agent behavior.

---

## Use Cases

- Agentic AI experimentation  
- Workflow automation prototypes  
- LangGraph / LangChain reference implementation  
- Human–AI collaboration systems  

---

## License

This project is intended for educational and experimental purposes.  
Ensure compliance with Google API usage policies before deploying in production.

