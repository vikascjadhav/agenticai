# Call Report Agent

LangGraph-based agent that:
- reads a call transcript from `call-reports.json`
- summarizes the call
- extracts action items
- calls a tool (`generate_action_items`) to simulate task creation
- logs node updates and state snapshots

## Project Files

- `call-report-agent.py`: main agent workflow
- `agent_data_models.py`: state and Pydantic models
- `call-reports.json`: input transcript data
- `requirements.txt`: Python dependencies

## Requirements

- Python 3.11+ (currently works on your environment; you may see a non-blocking warning on Python 3.14)
- Local OpenAI-compatible model endpoint (current code uses LM Studio style endpoint):
  - `http://localhost:1234/v1`
  - model: `google/gemma-3-4b`

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Run

```powershell
py .\call-report-agent.py
```

## How It Works

1. `summarize_call_transcript_node`
- Uses base LLM (no tools) to summarize transcript text.

2. `pending_actions_node`
- Prompts tool-bound LLM to call `generate_action_items` with strict schema:
  - `{"action_items":[{"description":"...","owner":"...","due_date":"..."}]}`

3. `tools` (`ToolNode`)
- Executes `generate_action_items`.

Graph edges:
- `START -> summarize_call_transcript_node -> pending_actions_node`
- conditional from `pending_actions_node` using `tools_condition`
- `tools -> END`

## Checkpointing (InMemorySaver)

This project uses:
- `store = InMemorySaver()`
- `app = agent.compile(checkpointer=store)`

When streaming/invoking with a checkpointer, pass config like:

```python
config = {"configurable": {"thread_id": "call-report-001"}}
```

Example:

```python
for mode, payload in app.stream(agent_state, config=config, stream_mode=["updates", "values"]):
    ...
```

To list checkpoints for a thread:

```python
for cp in store.list(config={"configurable": {"thread_id": "call-report-001"}}):
    print(cp)
```

## Logging

Current logs include:
- `agent.start`, `agent.end`
- node lifecycle and outputs
- tool results
- state snapshots (`stream_mode="values"`)

## Common Issues

1. `Checkpointer requires ... thread_id`
- Add `config={"configurable":{"thread_id":"..."}}` to `app.stream(...)` or `app.invoke(...)`.

2. Tool validation errors for action items
- Ensure model sends exact key names expected by schema (`due_date`, not `due date`).

3. Missing local model endpoint
- Start your local inference server at `http://localhost:1234/v1`.

