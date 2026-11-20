"""
CLI entrypoint for the Agentic Baseline.
Usage: python -m agent.cli --system-prompt "..." --user-prompt "..."
"""
import asyncio
import typer
from agent.wiring import build_agent_from_config
from agent.load_folder_yamls import load_folder_yamls, prune_to_matching_key

app = typer.Typer(help="Agentic Baseline CLI")


@app.command()
def run(
    system_prompt: str = typer.Option(..., help="System prompt for the agent"),
    user_prompt: str = typer.Option(..., help="User prompt to process"),
    session_id: str = typer.Option("default", help="Session identifier"),
) -> None:
    config = {
        'augmentation': {'path': 'layers/augmentation', 'key': 'functions'},
        'orchestration': {'path': 'layers/orchestration', 'key': 'llm'},
        'memory': {'path': 'layers/memory', 'key': 'context'},
        'evaluation': {'path': 'layers/evaluation', 'key': 'test'},
        'guardrails': {'path': 'layers/guardrails', 'key': 'policy'},
        'agent': {'path': '', 'key': 'harness'},
    }
    agent_config = {}
    for key in config.keys():
        agent_config.update(load_folder_yamls(config.get(key, None)))
    agent = build_agent_from_config(agent_config)
    result = asyncio.run(agent.run(session_id, system_prompt, user_prompt))
    print(result)


if __name__ == "__main__":
    app()