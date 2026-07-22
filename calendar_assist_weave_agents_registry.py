"""
Calendar assistant demo with Weave's Agents conversation/turn tracking.

This keeps Pydantic AI in charge of the agent loop, while Weave records the
multi-turn conversation and each local tool execution for the Agents view.
"""
import json
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
PROJECT = "pydanticai_demo"
WEAVE_PROJECT = f"{ENTITY}/{PROJECT}"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_DIR = BASE_DIR / "prompts"

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

    with wandb.init(entity=ENTITY, project=PROJECT, job_type="publish-code") as run:
        code_artifact = wandb.Artifact(name="calendar-assistant-code", type="code")
        code_artifact.add_dir(str(BASE_DIR / "tools"), name="tools")
        code_artifact.add_file(str(BASE_DIR / "calendar_assist_weave_agents_registry.py"))
        code_artifact.add_file(str(BASE_DIR / "requirements.txt"))
        logged_code_artifact = run.log_artifact(code_artifact)
        logged_code_artifact.wait() # Wait for the artifact to finish logging before proceeding

        # Store the artifact reference in the manifest for tracking purposes
        code_artifact_ref = logged_code_artifact.qualified_name

    # Initialize a W&B run, log and link the prompt artifact to the W&B Registry,
    # and store the artifact reference in the manifest for tracking purposes.
    # NOTE: This is the old way of logging artifacts to the W&B Registry.
    # The new way is to use the `weave.publish` + `weave.link_prompt_to_registry` methods below.
    with wandb.init(entity=ENTITY, project=PROJECT, job_type="publish-prompt") as run:
        used_code_artifact = run.use_artifact(code_artifact_ref, type="code")

        prompt_artifact = wandb.Artifact(
            name=manifest["name"],
            type=manifest.get("artifact_type", "prompt"),
            description=manifest.get("version_description"),
            metadata={**manifest, "code_artifact_ref": used_code_artifact.qualified_name},
        )
        prompt_artifact.add_dir(str(args.prompt_dir))

        linked_prompt = run.link_artifact(
            artifact=prompt_artifact,
            target_path=registry_target,
            aliases=["latest"],
        )

        # Wait for the linked prompt artifact to finish logging before proceeding
        linked_prompt.wait()

        # Store the artifact reference in the manifest for tracking purposes
        prompt_artifact_ref = linked_prompt.qualified_name

    # Initialize Weave, publish the prompt to Weave, and link it to the W&B Registry
    weave_client = weave.init(WEAVE_PROJECT)
    weave_prompt_ref = weave.publish(
        weave.StringPrompt(prompt),
        name=manifest["name"]
    )
    weave_client.link_prompt_to_registry(
        prompt=weave_prompt_ref,
        target_path=registry_target
    )

    # Build the calendar assistant agent with the loaded prompt and model
    calendar_agent = build_agent(model, agent_name, prompt)
    message_history: list[ModelMessage] | None = None

    with weave.start_conversation(
        agent_name=calendar_agent.agent_name,
        model=calendar_agent.model,
        conversation_name="calendar-assistant-demo",
        attributes={
            "prompt.name": manifest.get("name", ""),
            "prompt.artifact_type": manifest.get("artifact_type", ""),
            "prompt.artifact_ref": prompt_artifact_ref,
            "prompt.registry_target": registry_target,
            "prompt.weave_ref": weave_prompt_ref.uri,
            "code.artifact_ref": code_artifact_ref,
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
