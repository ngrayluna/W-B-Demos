"""
Calendar assistant demo with Weave's Agents conversation/turn tracking.

This keeps Pydantic AI in charge of the agent loop, while Weave records the
multi-turn conversation and each local tool execution for the Agents view.
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from tools.availability import find_available_slots
from typing import Any, Sequence

import weave
import wandb
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

ENTITY = "wandb"
PROJECT = "pydantic_demo"
WEAVE_PROJECT = f"{ENTITY}/{PROJECT}"

DEFAULT_PROMPT_DIR = Path(__file__).with_name("prompts")

@dataclass
class CalendarAgent:
    agent: Agent
    model: str
    agent_name: str
    prompt: str


def build_agent(model: str, agent_name: str, prompt: str) -> CalendarAgent:
    """Build the calendar assistant and register its local tools."""
    calendar_agent = Agent(
        model=model,
        output_type=str,
        name=agent_name,
        instructions=prompt,
    )
    calendar_agent.tool_plain(find_available_slots)
    return CalendarAgent(
        agent=calendar_agent,
        model=model,
        agent_name=agent_name,
        prompt=prompt,
    )


def run_turn(
    calendar_agent: CalendarAgent,
    user_input: str,
    message_history: Sequence[ModelMessage] | None = None,
) -> list[ModelMessage]:
    """Run one user turn and record it as a Weave Agents turn."""
    print(f"\nUser: {user_input}")

    # Wrap each Pydantic AI agent turn in a Weave Agents turn, so that the entire conversation
    # is tracked in the Weave Agents view.
    with weave.start_turn(
        user_message=user_input,
        model=calendar_agent.model,
        agent_name=calendar_agent.agent_name,
        system_instructions=[calendar_agent.prompt],
    ):
        result = calendar_agent.agent.run_sync(
            user_input,
            message_history=message_history,
        )

    print(f"Assistant: {result.output}")
    return list(result.all_messages())


def load_prompt_info(prompt_dir: Path) -> tuple[str, dict[str, Any]]:
    """Load the prompt text and manifest metadata from the prompt directory."""
    with (prompt_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    # Read the prompt text from the file specified in the manifest
    prompt_file = prompt_dir / manifest["prompt_file"]
    prompt = prompt_file.read_text(encoding="utf-8")

    return prompt, manifest


def main(args: Namespace) -> None:
    
    print("Calendar Assistant is running with Weave Agents tracking...")

    # Load the prompt and manifest metadata from the specified prompt directory
    prompt, manifest = load_prompt_info(args.prompt_dir)
    
    model = manifest.get("model")
    agent_name = manifest.get("agent_name")
    registry_target = f"wandb-registry-{manifest.get('registry')}/{manifest.get('collection')}"

    with wandb.init(entity = ENTITY, project=PROJECT, job_type="link calendar agent prompts to registry") as wandb_run:
        artifact = wandb.Artifact(
            name=manifest["name"],
            type=manifest.get("artifact_type", "prompt"),
            description=manifest.get("version_description"),
            metadata=manifest,
        )
        artifact.add_dir(str(args.prompt_dir))
        prompt_artifact = wandb_run.link_artifact(artifact=artifact, target_path=registry_target, aliases=["latest"])

        # Build the calendar assistant agent with the loaded prompt and model
        calendar_agent = build_agent(model, agent_name, prompt)
        message_history: list[ModelMessage] | None = None

        weave.init(WEAVE_PROJECT)
        with weave.start_conversation(
            agent_name=calendar_agent.agent_name,
            model=calendar_agent.model,
            conversation_name="calendar-assistant-demo",
            attributes={
                "prompt.name": manifest.get("name", ""),
                "prompt.artifact_type": manifest.get("artifact_type", ""),
                "prompt.artifact_ref": prompt_artifact.qualified_name,
                "prompt.registry_target": registry_target,
            },
        ):
            message_history = run_turn(
                calendar_agent,
                "I want to schedule a meeting next week. Can you help me find available time slots?",
                message_history,
            )

            message_history = run_turn(
                calendar_agent,
                "Wednesday afternoon for 30 minutes.",
                message_history,
            )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run the calendar assistant with Weave Agents tracking."
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=DEFAULT_PROMPT_DIR,
        help="Directory containing manifest.json and the prompt markdown file.",
    )
    args = parser.parse_args()
    main(args)
