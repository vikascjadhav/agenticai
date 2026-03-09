import json
import logging

from agent_data_models import (
    ActionItem,
    ActionItemRequest,
    ActionItemResult,
    AgentState,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_transcript() -> str:
    with open("call-reports.json", "r", encoding="utf-8") as f:
        call_transcripts = json.load(f)
    lines = call_transcripts[0]["transcript"]
    return "\n".join(lines)


def local_gemma_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="google/gemma-3-4b",
        api_key="fake-api-key",
    )


def summarize_call_transcript_node(state: AgentState):
    logger.info("node.start summarize_call_transcript_node")
    response = base_llm.invoke(
        f"Summarize the following call transcript:\n{state['text']}"
    )
    logger.info("node.end summarize_call_transcript_node")
    return {"summary": response.content, "messages": [response]}


def propose_action_items_node(state: AgentState):
    logger.info("node.start propose_action_items_node")
    extractor = base_llm.with_structured_output(ActionItemRequest)
    proposal = extractor.invoke(
        "Extract action items from the following transcript.\n"
        "Return only structured proposed_action_items.\n\n"
        f"Transcript:\n{state['text']}"
    )
    logger.info(
        "node.end propose_action_items_node proposed_count=%s",
        len(proposal.proposed_action_items),
    )
    return {
        "proposed_action_items": proposal.proposed_action_items,
        "approved_action_items": [],
        "created_tasks": [],
        "review_index_of_action_items": 0,
    }


def review_action_item_node(state: AgentState):
    logger.info("node.start review_action_item_node")
    idx = state["review_index_of_action_items"]
    items = state["proposed_action_items"]

    if idx >= len(items):
        logger.info("node.end review_action_item_node no_more_items")
        return {}

    item = items[idx]
    decision = interrupt(
        {
            "type": "review_action_item",
            "index": idx,
            "total": len(items),
            "item": item.model_dump(),
            "question": "Approve this action item? (y/n)",
        }
    )

    approved = list(state["approved_action_items"])
    if decision.get("approve", False):
        approved.append(item)
        logger.info("review.approved index=%s", idx)
    else:
        logger.info("review.rejected index=%s", idx)

    logger.info("node.end review_action_item_node")
    return {
        "approved_action_items": approved,
        "review_index_of_action_items": idx + 1,
    }


def create_approved_action_items_node(state: AgentState):
    logger.info("node.start create_approved_action_items_node")
    created_tasks: list[ActionItemResult] = []

    for item in state["approved_action_items"]:
        task = ActionItemResult(
            description=item.description,
            owner=item.owner,
            due_date=item.due_date,
            status="created",
        )
        created_tasks.append(task)
        logger.info(
            "task.create description=%s owner=%s due_date=%s",
            task.description,
            task.owner,
            task.due_date,
        )

    logger.info(
        "node.end create_approved_action_items_node created_count=%s",
        len(created_tasks),
    )
    return {"created_tasks": created_tasks}


def review_router(state: AgentState):
    if state["review_index_of_action_items"] < len(state["proposed_action_items"]):
        return "review_action_item_node"
    return "create_approved_action_items_node"


base_llm = local_gemma_model()
store = InMemorySaver()

agent = StateGraph(AgentState)
agent.add_node(summarize_call_transcript_node)
agent.add_node(propose_action_items_node)
agent.add_node(review_action_item_node)
agent.add_node(create_approved_action_items_node)

agent.add_edge(START, "summarize_call_transcript_node")
agent.add_edge("summarize_call_transcript_node", "propose_action_items_node")
agent.add_edge("propose_action_items_node", "review_action_item_node")
agent.add_conditional_edges(
    "review_action_item_node",
    review_router,
    {
        "review_action_item_node": "review_action_item_node",
        "create_approved_action_items_node": "create_approved_action_items_node",
    },
)
agent.add_edge("create_approved_action_items_node", END)

app = agent.compile(checkpointer=store)

graph = app.get_graph()


with open("graph.png", "wb") as f:
    f.write(graph.draw_mermaid_png())

def run_hitl_flow():
    logger.info("agent.start")
    config = {"configurable": {"thread_id": "call-report-001"}}

    initial_state: AgentState = {
        "text": load_transcript(),
        "summary": "",
        "proposed_action_items": [],
        "approved_action_items": [],
        "created_tasks": [],
        "review_index_of_action_items": 0,
        "messages": [],
    }

    next_input: dict | Command = initial_state
    while True:
        interrupted = False
        for event in app.stream(next_input, config=config, stream_mode="updates"):
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"][0].value
                item = interrupt_data["item"]
                logger.info(
                    "review.prompt index=%s/%s item=%s",
                    interrupt_data["index"] + 1,
                    interrupt_data["total"],
                    item,
                )
                answer = input(
                    f"Approve action item #{interrupt_data['index'] + 1} "
                    f"({item['description']})? [y/n]: "
                ).strip().lower()
                approved = answer in ("y", "yes")
                next_input = Command(resume={"approve": approved})
                interrupted = True
                break

        if not interrupted:
            break

    final_state = app.get_state(config).values
    logger.info("summary=%s", final_state.get("summary", ""))
    logger.info(
        "approved_count=%s created_count=%s",
        len(final_state.get("approved_action_items", [])),
        len(final_state.get("created_tasks", [])),
    )
    logger.info("agent.end")


if __name__ == "__main__":
    pass
    #run_hitl_flow()
