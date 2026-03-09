# Call Report Agent (LangGraph + HITL)

This project processes advisor-client call transcripts and demonstrates a production-style **human-in-the-loop (HITL)** workflow:

1. Select a transcript.
2. Generate a concise call summary.
3. Extract proposed action items.
4. Review each action item with approve/reject decisions.
5. Create tasks only for approved items.

The project includes:
- a **GUI app** (Streamlit)
- a **CLI/non-GUI app**
- strongly typed schemas for model boundaries
- checkpointed graph execution with LangGraph

## Project Structure

- `call-report-agent-with-gui.py`  
  Streamlit UI with transcript picker, processing indicators, review controls, state history, and created task table.

- `call-report-agent-without-gui.py`  
  Terminal flow for the same HITL graph pattern.

- `agent_data_models.py`  
  Data models and state schema (`TypedDict` for graph state, `Pydantic` for LLM structured outputs).

- `call-reports.json`  
  Input transcript dataset.

- `requirements.txt`  
  Python dependencies.

## Libraries Used

Core agent orchestration:
- `langgraph`  
  Graph orchestration, interrupts, checkpointing.
- `langchain-core`
- `langchain-openai`  
  OpenAI-compatible model client used against local endpoint.

Model/schema:
- `pydantic`  
  Structured outputs and boundary validation.

UI:
- `streamlit`  
  Interactive app UI.

Utilities:
- `python-dotenv` (available if you later externalize config)
- `langchain-google-genai` (currently not used in active files, but available)

## Architecture and Best-Practice Design

### 1) State Model Strategy

The graph state (`AgentState`) is a `TypedDict` and stores **primitive data** (`dict`, `list`, `str`) for stable checkpoint serialization.

Why:
- avoids checkpoint serde warnings
- keeps state portable and robust

### 2) Pydantic at Boundaries

Pydantic models are used at LLM/business boundaries:
- `ActionItemRequest` for structured extraction
- `ActionItemResult` for task creation validation

The app converts models to `dict` immediately before storing in graph state.

### 3) HITL via Interrupts

The review node calls `interrupt(...)` per action item.
UI/CLI resumes graph with `Command(resume={"approve": True|False})`.

### 4) Checkpointing

Graph uses `InMemorySaver` checkpointer.
Each run has unique `thread_id` to isolate state history.

## Graph Flow

`START`  
-> `summarize_call_transcript_node`  
-> `propose_action_items_node`  
-> `review_action_item_node` (loop until all reviewed)  
-> `create_approved_action_items_node`  
-> `END`

## Setup

## Prerequisites

- Python 3.11+ (works on your environment; Python 3.14 may show non-blocking warnings from dependencies)
- Local OpenAI-compatible model endpoint running at:
  - `http://localhost:1234/v1`
  - model id currently configured: `google/gemma-3-4b`

## Install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Run (GUI)

```powershell
.\.venv\Scripts\streamlit.exe run .\call-report-agent-with-gui.py
```

What you’ll see:
- left/sidebar:
  - transcript selector
  - start run button
  - graph visualization (image if available)
- main:
  - summary
  - proposed items table
  - created tasks table
  - review queue (approve/reject)
  - selected transcript panel (right side)
  - state history table + state inspector

## Run (Without GUI)

```powershell
.\.venv\Scripts\python.exe .\call-report-agent-without-gui.py
```

You’ll be prompted in terminal for each action item approval.

## Troubleshooting

### 1) Streamlit not found

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2) Local model connection fails (`http://localhost:1234/v1`)

Make sure local model server (LM Studio / compatible gateway) is running and model is loaded.

### 3) Graph image not available

The GUI attempts graph image rendering. In restricted environments this can fail.
The app handles this gracefully and continues without crashing.

### 4) Follow-up sentence in summary

Summary prompt is constrained to avoid follow-up questions/offers. If model still drifts:
- tighten prompt rules
- reduce temperature
- add output post-processing guardrails

## Customization

Common changes:
- switch model in `local_gemma_model()`
- change transcript source (`call-reports.json` loader)
- extend action item schema (priority, category, system ticket id)
- replace mock task creation with real API calls

## Notes on Code Quality

Current implementation follows:
- boundary conversion for structured outputs
- interrupt/resume HITL pattern
- state history visibility for debugging
- explicit node routing and conditional edges

