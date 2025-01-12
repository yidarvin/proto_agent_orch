import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

def format_claude_content(response):
    final_text = []
    for content in response.content:
        if content.type == 'text':
            final_text.append(content.text)
        else:
            continue
    return "\n".join(final_text)

class SONNETagent:
    def __init__(self):
        self.model = "claude-3-5-sonnet-20241022"
        self.max_tokens=1000
        self.messages = []
        self.anthropic = Anthropic()
    
    def execute(self, text):
        self.messages.append(
            {
                "role": "user",
                "content": text
            }
        )
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self.messages,
        )
        return format_claude_content(response)

class HAIKUagent:
    def __init__(self):
        self.model = "claude-3-5-haiku-20241022"
        self.max_tokens=1000
        self.messages = []
        self.anthropic = Anthropic()
    
    def execute(self, text):
        self.messages.append(
            {
                "role": "user",
                "content": text
            }
        )
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self.messages,
        )
        return format_claude_content(response)

class TEXTagent:
    def __init__(self):
        return None

    def execute(self, text):
        return text

def main():
    sa = SONNETagent()
    ha = HAIKUagent()
    ta = TEXTagent()

    try:
        print(sa.execute("What's 2 + 2?"))
        print(ha.execute("What's 2 + 2?"))
        print(ta.execute("This should be text."))
    finally:
        print("That's all folks.")

if __name__ == "__main__":
    main()