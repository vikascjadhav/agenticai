"""Streamlit UI for human-in-the-loop call report processing.

Flow:
1. User selects a transcript.
2. Agent summarizes the transcript.
3. Agent proposes action items.
4. User approves/rejects each item one-by-one.
5. Only approved items are logged as created tasks.
"""

import json
import logging
from pathlib import Path
from uuid import uuid4

import streamlit as st
from agent_data_models import ActionItemRequest, ActionItemResult, AgentState
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CALL_REPORTS_PATH = Path(__file__).with_name("call-reports.json")


def load_call_reports() -> list[dict]:
    """Load transcript records from local JSON."""
    with CALL_REPORTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def transcript_to_text(report: dict) -> str:
    """Flatten transcript lines into one text block for the model."""
    return "\n".join(report["transcript"])


def local_gemma_model() -> ChatOpenAI:
    """Create OpenAI-compatible local model client (LM Studio/Ollama gateway)."""
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="google/gemma-3-4b",
        api_key="fake-api-key",
    )


def state_view(state: dict) -> dict:
    """Return a UI-safe state projection (no heavy internals)."""
    return {
        "summary": state.get("summary", ""),
        "review_index_of_action_items": state.get("review_index_of_action_items", 0),
        "proposed_action_items": state.get("proposed_action_items", []),
        "approved_action_items": state.get("approved_action_items", []),
        "created_tasks": state.get("created_tasks", []),
    }


@st.cache_resource
def build_agent():
    """Build and cache LangGraph app + optional graph image."""
    base_llm = local_gemma_model()
    store = InMemorySaver()

    def summarize_call_transcript_node(state: AgentState):
        # The prompt explicitly forbids follow-up prompts/questions.
        response = base_llm.invoke(
            "You are generating an internal call report summary.\n"
            "Output rules:\n"
            "- Provide only the summary text.\n"
            "- Do not ask follow-up questions.\n"
            "- Do not add offers like 'Would you like...'.\n"
            "- Keep it concise and factual (4-6 sentences).\n\n"
            f"Transcript:\n{state['text']}"
        )
        return {"summary": response.content, "messages": [response]}

    def propose_action_items_node(state: AgentState):
        # Use structured output for robust extraction, then persist primitives.
        extractor = base_llm.with_structured_output(ActionItemRequest)
        proposal = extractor.invoke(
            "Extract action items from the following transcript.\n"
            "Return only structured proposed_action_items.\n\n"
            f"Transcript:\n{state['text']}"
        )
        proposed_items = [item.model_dump() for item in proposal.proposed_action_items]
        return {
            "proposed_action_items": proposed_items,
            "approved_action_items": [],
            "created_tasks": [],
            "review_index_of_action_items": 0,
        }

    def review_action_item_node(state: AgentState):
        # Pause graph execution and wait for human decision via interrupt.
        idx = state["review_index_of_action_items"]
        items = state["proposed_action_items"]
        if idx >= len(items):
            return {}

        item = items[idx]
        decision = interrupt(
            {
                "type": "review_action_item",
                "index": idx,
                "total": len(items),
                "item": item,
                "question": "Approve this action item?",
            }
        )

        approved = list(state["approved_action_items"])
        if decision.get("approve", False):
            approved.append(item)
        return {
            "approved_action_items": approved,
            "review_index_of_action_items": idx + 1,
        }

    def create_approved_action_items_node(state: AgentState):
        # Boundary validation: ensure created tasks conform to response schema.
        created_tasks: list[dict] = []
        for item in state["approved_action_items"]:
            task = ActionItemResult(
                description=item["description"],
                owner=item["owner"],
                due_date=item["due_date"],
                status="created",
            )
            created_tasks.append(task.model_dump())
        return {"created_tasks": created_tasks}

    def review_router(state: AgentState):
        # Keep looping review until every proposed item has a decision.
        if state["review_index_of_action_items"] < len(state["proposed_action_items"]):
            return "review_action_item_node"
        return "create_approved_action_items_node"

    graph = StateGraph(AgentState)
    graph.add_node(summarize_call_transcript_node)
    graph.add_node(propose_action_items_node)
    graph.add_node(review_action_item_node)
    graph.add_node(create_approved_action_items_node)

    graph.add_edge(START, "summarize_call_transcript_node")
    graph.add_edge("summarize_call_transcript_node", "propose_action_items_node")
    graph.add_edge("propose_action_items_node", "review_action_item_node")
    graph.add_conditional_edges(
        "review_action_item_node",
        review_router,
        {
            "review_action_item_node": "review_action_item_node",
            "create_approved_action_items_node": "create_approved_action_items_node",
        },
    )
    graph.add_edge("create_approved_action_items_node", END)

    app = graph.compile(checkpointer=store)

    # Prefer graph image for UI visualization; fallback handled by caller.
    graph_png = None
    try:
        try:
            graph_png = app.get_graph().draw_mermaid_png(max_retries=1)
        except TypeError:
            graph_png = app.get_graph().draw_mermaid_png()
    except Exception as e:
        logger.warning("Graph image unavailable: %s", e)

    return app, graph_png


def run_until_pause_or_end(app, next_input, config: dict, progress_cb=None):
    """Run graph until either an interrupt occurs or execution reaches end."""
    interrupt_payload = None
    history_events: list[dict] = []
    step = 0

    for event in app.stream(next_input, config=config, stream_mode="updates"):
        step += 1
        if "__interrupt__" in event:
            interrupt_payload = event["__interrupt__"][0].value
            history_events.append(
                {
                    "event": "interrupt",
                    "step": step,
                    "interrupt_index": interrupt_payload["index"],
                    "interrupt_total": interrupt_payload["total"],
                }
            )
            if progress_cb:
                progress_cb(step, "interrupt")
            break

        nodes = []
        for node, output in event.items():
            if node == "__interrupt__":
                continue
            keys = list(output.keys()) if isinstance(output, dict) else []
            nodes.append({"node": node, "keys": keys})
            if progress_cb:
                progress_cb(step, node)

        if nodes:
            history_events.append({"event": "node_update", "step": step, "nodes": nodes})

    state = app.get_state(config).values
    snapshot = state_view(state)
    done = interrupt_payload is None and (
        snapshot["review_index_of_action_items"] >= len(snapshot["proposed_action_items"])
    )
    history_events.append({"event": "state_snapshot", "step": step + 1, "state": snapshot})
    return state, interrupt_payload, done, history_events


def initialize_state():
    """Initialize all Streamlit session keys used by the UI."""
    defaults = {
        "thread_id": "",
        "config": {},
        "agent_state": {},
        "pending_interrupt": None,
        "is_complete": False,
        "state_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_history(action: str, events: list[dict]):
    """Append execution events to history for traceability/debugging."""
    for event in events:
        st.session_state.state_history.append(
            {
                "index": len(st.session_state.state_history) + 1,
                "action": action,
                **event,
            }
        )


def main():
    st.set_page_config(
        page_title="Call Report Agent",
        page_icon=":telephone_receiver:",
        layout="wide",
    )
    st.title("Call Report Agent (Human-in-the-Loop)")
    st.caption(
        "Select a transcript, process with the agent, review action items one-by-one, and create approved tasks."
    )

    initialize_state()
    app, graph_png = build_agent()
    reports = load_call_reports()

    with st.sidebar:
        st.subheader("Input")
        options = [
            f"{idx + 1}. {report.get('client_name', f'Client {idx + 1}')}"
            for idx, report in enumerate(reports)
        ]
        selected_label = st.selectbox("Select transcript", options=options, index=0)
        selected_idx = options.index(selected_label)
        selected_report = reports[selected_idx]
        selected_transcript = transcript_to_text(selected_report)

        if st.button("Start New Agent Run", type="primary", use_container_width=True):
            transcript = selected_transcript
            thread_id = f"call-report-{uuid4()}"
            config = {"configurable": {"thread_id": thread_id}}
            initial_state: AgentState = {
                "text": transcript,
                "summary": "",
                "proposed_action_items": [],
                "approved_action_items": [],
                "created_tasks": [],
                "review_index_of_action_items": 0,
                "messages": [],
            }

            status = st.status("Processing transcript...", expanded=True)
            progress = st.progress(0)

            def progress_cb(step, node_name):
                pct = min(90, 20 + step * 25)
                progress.progress(pct)
                if node_name:
                    status.write(f"Processed: `{node_name}`")

            state, pending_interrupt, done, events = run_until_pause_or_end(
                app, initial_state, config, progress_cb
            )
            progress.progress(100)
            status.update(label="Initial processing complete", state="complete", expanded=False)

            st.session_state.thread_id = thread_id
            st.session_state.config = config
            st.session_state.agent_state = state
            st.session_state.pending_interrupt = pending_interrupt
            st.session_state.is_complete = done
            st.session_state.state_history = []
            add_history("start_run", events)

        st.divider()
        st.subheader("Graph")
        if graph_png:
            st.image(graph_png, caption="LangGraph flow", use_container_width=True)
        else:
            st.caption("Graph image unavailable in this environment.")

    state = st.session_state.agent_state or {}
    pending_interrupt = st.session_state.pending_interrupt
    state_snapshot = state_view(state)

    col1, col2 = st.columns([3, 2])

    with col2:
        st.subheader("Selected Transcript")
        st.caption(
            f"Client: {selected_report.get('client_name', f'Client {selected_idx + 1}')}"
        )
        st.text_area(
            "Transcript",
            value=selected_transcript,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    if not st.session_state.config:
        with col1:
            st.info("Start a new run from the left sidebar.")
        return

    with col1:
        st.subheader("Call Summary")
        st.write(state_snapshot.get("summary") or "Summary not generated yet.")

        st.subheader("Proposed Action Items")
        proposed = state_snapshot.get("proposed_action_items", [])
        if proposed:
            st.dataframe(proposed, use_container_width=True)
        else:
            st.caption("No proposed action items yet.")

        st.subheader("Created Action Items")
        created = state_snapshot.get("created_tasks", [])
        if created:
            st.dataframe(created, use_container_width=True)
        else:
            st.caption("No created tasks yet.")

    with col2:
        st.subheader("Review Queue")
        if pending_interrupt and not st.session_state.is_complete:
            item = pending_interrupt["item"]
            st.markdown(
                f"**Item {pending_interrupt['index'] + 1} of {pending_interrupt['total']}**"
            )
            st.json(item)
            decision_key = f"review_decision_{pending_interrupt['index']}"
            decision = st.radio(
                "Decision",
                options=["Approve", "Reject"],
                horizontal=True,
                key=decision_key,
            )
            if st.button("Submit Decision", type="primary", use_container_width=True):
                status = st.status("Applying decision...", expanded=True)
                progress = st.progress(0)

                def progress_cb(step, node_name):
                    pct = min(90, 25 + step * 30)
                    progress.progress(pct)
                    if node_name:
                        status.write(f"Processed: `{node_name}`")

                state, pending_interrupt, done, events = run_until_pause_or_end(
                    app,
                    Command(resume={"approve": decision == "Approve"}),
                    st.session_state.config,
                    progress_cb,
                )
                progress.progress(100)
                status.update(label="Decision processed", state="complete", expanded=False)

                st.session_state.agent_state = state
                st.session_state.pending_interrupt = pending_interrupt
                st.session_state.is_complete = done
                add_history(f"decision:{decision.lower()}", events)
                st.rerun()
        else:
            if st.session_state.is_complete:
                st.success("Review complete. Approved items are created in task management.")
            else:
                st.info("No pending review.")

        st.subheader("Run Details")
        st.write(f"Thread ID: `{st.session_state.thread_id}`")
        st.write(
            f"Approved: **{len(state_snapshot.get('approved_action_items', []))}** | "
            f"Created: **{len(state_snapshot.get('created_tasks', []))}**"
        )

    st.subheader("Agent State History")
    if st.session_state.state_history:
        rows = []
        for h in st.session_state.state_history:
            snapshot = h.get("state", {})
            nodes = ", ".join([n["node"] for n in h.get("nodes", [])]) if h.get("nodes") else ""
            rows.append(
                {
                    "index": h["index"],
                    "action": h["action"],
                    "event": h["event"],
                    "step": h["step"],
                    "nodes": nodes,
                    "approved": len(snapshot.get("approved_action_items", [])),
                    "created": len(snapshot.get("created_tasks", [])),
                }
            )
        st.dataframe(rows, use_container_width=True)

        selected_idx = st.number_input(
            "Inspect history entry",
            min_value=1,
            max_value=len(st.session_state.state_history),
            value=len(st.session_state.state_history),
            step=1,
        )
        st.json(st.session_state.state_history[selected_idx - 1])
    else:
        st.caption("No history yet.")

    with st.expander("Current Agent State", expanded=False):
        st.json(state_snapshot)


if __name__ == "__main__":
    main()
