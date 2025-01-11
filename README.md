# proto_agent_orch
Test code for a quick and dirty prototype of an agent propose/revision orchestrator.

# Requirements for macOS
`python >= 3.10`
`brew install uv`

# Setting Up Client
Add a .env file with your API key in orchestrator.  Standard `ANTHROPIC_API_KEY=<your key here>`

# UV Stuff
`uv init orchestrator`
`cd orchestrator`
`uv venv`
`source .venv/bin/activate`
`uv add mcp anthropic python-dotenv httpx`

# Running
`uv run orchestrator/client.py weather.py`