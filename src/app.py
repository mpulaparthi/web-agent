from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_core.messages import HumanMessage
from src.agent import run_agent

# Initialize Bedrock AgentCore Application
app = BedrockAgentCoreApp()

@app.entrypoint
async def agent_entrypoint(event):
    """
    Entrypoint for the Bedrock AgentCore Runtime.
    The event contains the user prompt and context.
    """
    print(f"Received event: {event}")
    
    # Extract user input
    # Depending on the event structure, it might be in 'prompt' or 'inputText'
    # The documentation example used payload.get("prompt")
    user_input = event.get("prompt") or event.get("inputText")
    
    if not user_input:
        return {"error": "No input provided"}
        
    print(f"Processing input: {user_input}")
    
    # Invoke the agent
    try:
        response = await run_agent(user_input)
        return {"response": response}
    except Exception as e:
        print(f"Error running agent: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    app.run()