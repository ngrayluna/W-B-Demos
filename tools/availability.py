import json
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
import weave

TIME_SLOTS_PATH = Path(__file__).parent.parent / "time_slots.json"

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

AVAILABLE_TIME_SLOTS = load_available_time_slots()

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
