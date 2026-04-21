# AutoStream: Social-to-Lead Agentic Workflow

This agent is built to identify high-intent leads, answer product questions via RAG, and store lead details in a local database.

## 1. How to Run Locally

1. **Setup**:
   ```bash
   pip install -r requirements.txt
Environment:

Add OPENROUTER_API_KEY=your_key to a .env file.

Run:

Bash
python main.py

View Leads:

After a successful lead capture, a leads.db file will be created along with faiss index folder. You can view the stored leads using any SQLite viewer. 

Architecture Explanation (≈200 words):

I chose LangGraph for this architecture because it provides a state-first approach to conversation design, which is essential for multi-turn lead qualification. Unlike linear chains, LangGraph allows the agent to maintain a "reasoning loop," ensuring the agent can capture missing data points even if the user asks an unrelated question in between.

1.State Management:

The state is managed through a global AgentState dictionary that tracks messages and a user_data object. By using a MemorySaver checkpointer, the agent achieves turn-persistence, allowing it to "remember" the user's name while it asks for their email. 
The final tool execution is strictly conditional: 
it only triggers the SQLite mock_lead_capture function once the state confirms that the name, email, and platform are all present and validated.


2.WhatsApp Deployment 

To integrate this agent with WhatsApp using Webhooks:

Host API: Deploy the agent using FastAPI to an HTTPS-enabled server.

Webhook Registration: Use the Meta for Developers portal to point WhatsApp Webhooks to your /chat endpoint.

Session ID Mapping: Use the user's phone number as the thread_id in LangGraph to ensure each WhatsApp user has their own private state memory.

Data Handling: When the Webhook receives a message, parse the JSON, pass the text to the agent, and use the WhatsApp Cloud API to send the agent's response back to the user.
