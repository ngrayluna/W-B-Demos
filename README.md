# Calendar Assistant — Weave Agents + W&B Registry

A small [Pydantic AI](https://ai.pydantic.dev/) calendar assistant that finds open
meeting slots, instrumented with [Weave](https://weave-docs.wandb.ai/) and the
[W&B Registry](https://docs.wandb.ai/guides/registry/).

The tracked demos do two things:

1. **Publish** the tool code and prompt as **W&B Artifacts**, then link the
   prompt to the **W&B Registry** (the code artifact is referenced for lineage).
2. **Track** the multi-turn conversation and every local tool call in
   **Weave's Agents view**.

Pydantic AI stays in charge of the agent loop; Weave wraps each turn and tool call.

## Layout

| File | What it is |
| --- | --- |
| `entry.py` | Minimal version — just the agent, tool, and a two-turn conversation. No tracking. |
| `calendar_assist_weave_agents_registry.py` | Full pipeline: publish artifacts, link to Registry, run the tracked conversation. Reads `prompts/` and `tools/`. |
| `calendar_assistant_molab.py` | Self-contained [marimo](https://marimo.io/) notebook version of the above; all data inlined so it runs anywhere (including molab). |
| `tools/availability.py` | The `find_available_slots` tool. |
| `prompts/` | `prompt.md` and `manifest.json` (model, agent name, registry target). |
| `time_slots.json` | Sample open slots the tool searches. |

## Requirements

```
pydantic-ai-slim[openai]
wandb
weave
```

Set credentials before running:

```bash
export OPENAI_API_KEY=...
export WANDB_API_KEY=...
```

## Run

```bash
# Minimal, no tracking
python entry.py

# Full pipeline (artifacts + Registry + Weave)
python calendar_assist_weave_agents_registry.py --prompt-dir prompts

# Notebook edition
marimo edit calendar_assistant_molab.py
```

After a tracked run, open the **Agents** view in your Weave project
(`wandb/pydanticai_demo` by default) to see the conversation and tool spans.
