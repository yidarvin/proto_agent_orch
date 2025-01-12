import ast

import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

from agents import SONNETagent, HAIKUagent, TEXTagent

load_dotenv()  # load environment variables from .env

ORCHESTRATOR_PROMPT = """
You have the following agents: SONNETagent (a strong LLM), HAIKUagent (a fast LLM), and TEXTagent (repeats back text).

You also have additional tools that have been passed into you.

Choose which agent to use, what text to pass into the agent, which tools to use (if any), and the arguments to those tools.

The output should be a dictionary of the form:
{
    agent: <agent to use>,
    agent_text: <text to pass to agent>,
    tools: <list of (tool, args) tuples>
}

Only give outputs in the form of this dictionary, no additional text.
"""

def format_claude_content(response):
    final_text = []
    for content in response.content:
        if content.type == 'text':
            final_text.append(content.text)
        else:
            continue
    return "\n".join(final_text)

class Orchestrator:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
    
    async def propose_actions(self, history) -> str:
        "Proposes actions: an agent to use, input to the agent, and tools/args."

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=history,
            tools=available_tools
        )
        #print(history)
        #print(response)

        return format_claude_content(response)
    
    async def execute_actions(self, actions):
        action_dict = ast.literal_eval(actions)
        if action_dict["agent"] == "SONNETagent":
            agent = SONNETagent()
        elif action_dict["agent"] == "HAIKUagent":
            agent = HAIKUagent()
        elif action_dict["agent"] == "TEXTagent":
            agent = TEXTagent()
        else:
            agent = TEXTagent()
        text = action_dict["agent_text"]
        for tuple_ta in action_dict["tools"]:
            tool_name, tool_args = tuple_ta
            result = await self.session.call_tool(tool_name, tool_args)
            text += '\n' + format_claude_content(result)
        return agent.execute(text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your initial query or 'quit' to exit.")
        query = input("\nQuery: ").strip()
        history = [
            {
                "role": "user",
                "content": query + ORCHESTRATOR_PROMPT
            }
        ]
        
        refinements = "execute"
        print("We will now propose actions.  Describe refinements or say execute.")
        while query.lower() != 'quit':
            try:
                response = await self.propose_actions(history)
                history.append({
                    "role": "assistant",
                    "content": response
                })
                print("\n" + response)
                refinement = input("\nDescribe refinements or say execute: ").strip()
                history.append({
                    "role": "user",
                    "content": refinement
                })
                if refinement.lower() == 'execute':
                    break
            except Exception as e:
                print(f"\nError: {str(e)}")
                break
        final = await self.execute_actions(response)
        print(history)
        print("\n\n----------\nFinal Result\n----------\n\n")
        print(final)
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <path_to_server_script>")
        sys.exit(1)
        
    client = Orchestrator()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())