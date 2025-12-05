from typing import List, Tuple, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic

from deepagents.agent import create_deep_agent
from deepagents.tools.task import TaskTool
from deepagents.tools.file import ReadFileTool, WriteFileTool

from tools import tavily_search_tool, api_call_tool
from prompts import DATA_EXTRACTION_AGENT_INSTRUCTIONS, SUB_AGENT_RESEARCH_PROMPT, FINAL_REPORT_PROMPT

def create_data_extraction_agent(
    model: ChatAnthropic,
    tools: List[Any],
    workflow_instructions: str,
    sub_agents: Dict[str, Any] = {},
) -> Any:
    """
    Creates a data extraction agent.
    """
    # The prompt will be dynamically formatted with available tools by the agent harness
    prompt = PromptTemplate.from_template(workflow_instructions)
    
    agent = create_deep_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        sub_agents=sub_agents,
    )
    return agent

if __name__ == "__main__":
    # Initialize the model (using Anthropic as per the deepagents example)
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)

    # Define the tools available to the agent
    data_extraction_tools = [
        tavily_search_tool,
        api_call_tool,
        ReadFileTool(),  # Built-in deepagents tool
        WriteFileTool(), # Built-in deepagents tool
        TaskTool()       # Built-in deepagents tool for sub-agent delegation
    ]

    # Define a simple sub-agent for research if needed
    research_sub_agent_definition = {
        "name": "research_sub_agent",
        "description": "A sub-agent specialized in performing web research.",
        "instructions": SUB_AGENT_RESEARCH_PROMPT,
        "tools": [tavily_search_tool],
        "model": llm # Sub-agent can use the same model or a different one
    }

    # Create the agent
    data_extraction_agent = create_data_extraction_agent(
        model=llm,
        tools=data_extraction_tools,
        workflow_instructions=DATA_EXTRACTION_AGENT_INSTRUCTIONS,
        sub_agents={"research_sub_agent": research_sub_agent_definition}
    )

    print("Data Extraction Agent created. You can now interact with it.")
    print("\n--- Example Interaction ---")
    
    # Example usage
    user_query = "Find the current stock price of Google (GOOGL) and write it to a file named 'google_stock.txt'."
    print(f"User Query: {user_query}")
    
    # The agent will process this query, potentially using tavily_search_tool
    # and then WriteFileTool.
    # Note: For this to work, TAVILY_API_KEY must be set in your environment.
    # The actual execution will happen in a Codespace.
    
    # You would typically interact with the agent like this in a LangGraph app:
    # from langgraph.graph import StateGraph, START
    # app = data_extraction_agent
    # app.invoke({"messages": [HumanMessage(content=user_query)]})
    
    print("\nTo run this agent, you would typically integrate it into a LangGraph application.")
    print("For local testing, you can modify `if __name__ == '__main__':` block to invoke the agent directly.")
    print("Example: data_extraction_agent.invoke({'messages': [HumanMessage(content=user_query)]})")
    print("Remember to set the TAVILY_API_KEY environment variable.")