"""
Calendar assistant demo with Weave's Agents conversation/turn tracking.

This keeps Pydantic AI in charge of the agent loop, while Weave records the
multi-turn conversation and each local tool execution for the Agents view.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import weave
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

MODEL = "openai:gpt-4o-mini"
AGENT_NAME = "Calendar Assistant"
WEAVE_PROJECT = os.getenv("WANDB_WEAVE_PROJECT", "wandb/pydantic_demo")

PROMPT = """
You are a calendar assistant that helps users find available time slots.

If the user has not provided enough scheduling details, ask a concise follow-up question.
Useful details are:
- day of week
- time of day
- meeting duration

Once you have enough details, call find_available_slots.
After the tool returns slots, summarize the best options clearly.
"""
TIME_SLOTS_PATH = Path(__file__).with_name("time_slots.json")


class AvailabilitySearchInput(BaseModel):
    day_of_week: str | None = Field(
        default=None,
        description="The day of the week for which the user wants to find available time slots.",
    )
    time_of_day: str | None = Field(
        default=None,
        description="The time of day for which the user wants to find available time slots.",
    )
    duration_minutes: int = Field(
        default=30,
        gt=0,
        description="The duration of the meeting in minutes.",
    )


def load_available_time_slots(path: Path = TIME_SLOTS_PATH) -> dict[str, list[str]]:
    """Load available slots from disk, grouped by uppercase weekday."""
    with path.open(encoding="utf-8") as slot_file:
        raw_slots = json.load(slot_file)["open_slots"]

    slots_by_day: dict[str, list[str]] = {}
    for slot in raw_slots:
        slot_time = datetime.fromisoformat(slot)
        day = slot_time.strftime("%A").upper()
        time = slot_time.strftime("%H:%M")
        slots_by_day.setdefault(day, []).append(time)

    return slots_by_day


AVAILABLE_TIME_SLOTS = load_available_time_slots()


def matches_time_of_day(slot_time: str, time_of_day: str | None) -> bool:
    if time_of_day is None or time_of_day.lower() == "any":
        return True

    hour = int(slot_time.split(":", maxsplit=1)[0])
    preference = time_of_day.lower()

    if preference == "morning":
        return 6 <= hour < 12

    if preference == "afternoon":
        return 12 <= hour < 17

    if preference == "evening":
        return 17 <= hour < 21

    return True


def find_available_slots(
    day_of_week: str | None = None,
    time_of_day: str | None = None,
    duration_minutes: int = 30,
) -> list[str]:
    """Find available time slots for a meeting."""
    arguments = {
        "day_of_week": day_of_week,
        "time_of_day": time_of_day,
        "duration_minutes": duration_minutes,
    }

    # Add the local Pydantic AI tool record as a Weave tool span
    with weave.start_tool(
        name="find_available_slots",
        arguments=json.dumps(arguments),
    ) as tool:
        search = AvailabilitySearchInput(**arguments)

        if search.day_of_week is None:
            days_to_search = AVAILABLE_TIME_SLOTS.keys()
        else:
            day = search.day_of_week.upper()
            if day not in AVAILABLE_TIME_SLOTS:
                tool.result = []
                return tool.result
            days_to_search = [day]

        slots = []
        for day in days_to_search:
            for time in AVAILABLE_TIME_SLOTS[day]:
                if matches_time_of_day(time, search.time_of_day):
                    slots.append(
                        f"{day} at {time} for {search.duration_minutes} minutes"
                    )

        tool.result = slots
        return slots


def build_agent() -> Agent:
    """Build the calendar assistant and register its local tools."""
    calendar_agent = Agent(
        model=MODEL,
        output_type=str,
        name=AGENT_NAME,
        instructions=PROMPT,
    )
    calendar_agent.tool_plain(find_available_slots)
    return calendar_agent


def run_turn(
    agent: Agent,
    user_input: str,
    message_history: Sequence[ModelMessage] | None = None,
) -> list[ModelMessage]:
    """Run one user turn and record it as a Weave Agents turn."""
    print(f"\nUser: {user_input}")

    # Wrap each Pydantic AI agent turn in a Weave Agents turn, so that the entire conversation
    # is tracked in the Weave Agents view.
    with weave.start_turn(
        user_message=user_input,
        model=MODEL,
        agent_name=AGENT_NAME,
        system_instructions=[PROMPT],
    ):
        result = agent.run_sync(user_input, message_history=message_history)

    print(f"Assistant: {result.output}")
    return list(result.all_messages())


def ensure_openai_api_key() -> None:
    """Exit before running the demo if OpenAI credentials are unavailable."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running the demo.")


def main() -> None:
    ensure_openai_api_key()
    weave.init(WEAVE_PROJECT)

    print("Calendar Assistant is running with Weave Agents tracking...")

    agent = build_agent()
    message_history: list[ModelMessage] | None = None

    with weave.start_conversation(
        agent_name=AGENT_NAME,
        model=MODEL,
        conversation_name="calendar-assistant-demo",
    ):
        message_history = run_turn(
            agent,
            "I want to schedule a meeting next week. Can you help me find available time slots?",
            message_history,
        )

        message_history = run_turn(
            agent,
            "Wednesday afternoon for 30 minutes.",
            message_history,
        )


if __name__ == "__main__":
    main()
