import os
import asyncio
import nest_asyncio
from typing import Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from browser_use import Agent as BrowserAgent, Browser
from bedrock_agentcore.tools.browser_client import browser_session

# Apply nest_asyncio to support running asyncio loops within existing loops
nest_asyncio.apply()

# Initialize Bedrock LLM
# Note: Ensure AWS credentials and region are configured in the environment
llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs={"temperature": 0.0}
)

@tool
async def browse_web(task: str):
    """
    Use a web browser to perform a task. 
    The task should be a clear instruction of what to do on the web.
    """
    print(f"Browser Tool called with task: {task}")
    
    # Check for Invesco Vision credentials
    vision_email = os.environ.get("VISION_EMAIL")
    vision_password = os.environ.get("VISION_PASSWORD")
    
    # If the task involves vision.invesco.com and credentials are provided, append them
    # This is a basic way to provide credentials to the agent.
    # In a production environment, you might want more sophisticated secret management.
    if vision_email and vision_password and ("vision.invesco.com" in task or "Invesco" in task):
        print("Injecting Invesco Vision credentials into task context.")
        task += f"\n\nIf you need to log in to vision.invesco.com, use the following credentials:\nEmail: {vision_email}\nPassword: {vision_password}\nDo not output the password in your final response."

    # Get AWS region from environment or default
    region = os.environ.get("AWS_REGION", "us-west-2")
    
    # Use Bedrock Agent Core Browser Tool
    async with browser_session(region) as client:
        # Generate CDP URL and headers
        ws_url, headers = client.generate_ws_headers()
        print(f"Connecting to browser session at {ws_url} in region {region}")
        
        # Initialize Browser with remote CDP connection
        browser = Browser(cdp_url=ws_url, headers=headers)
        
        agent = BrowserAgent(
            task=task,
            llm=llm,
            browser=browser
        )
        
        # Run the agent
        try:
            history = await agent.run()
            
            # Summarize the result
            result = history.final_result()
            if not result:
                # If no result, try to get the last action output
                return "Task completed (no final result returned explicitly)."
            return result
        except Exception as e:
            print(f"Error during browser session: {e}")
            raise e
        finally:
            # Ensure the browser connection is closed
            # The session itself is managed by the browser_session context manager
            await browser.close()

# Define the graph state
class AgentState(dict):
    messages: List[BaseMessage]

# Define the graph
workflow = StateGraph(AgentState)

# Bind tools to the LLM
tools = [browse_web]
llm_with_tools = llm.bind_tools(tools)

def call_model(state):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

app = workflow.compile()

async def run_agent(input_text: str):
    """
    Run the agent with the given input text.
    """
    inputs = {"messages": [HumanMessage(content=input_text)]}
    final_state = await app.ainvoke(inputs)
    return final_state["messages"][-1].content