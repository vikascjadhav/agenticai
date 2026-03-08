import json
import logging
from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode
from agent_data_models import AgentState, ActionItem, ActionItemRequest, ToolResponse, ActionItemResult
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import InMemorySaver


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

store = InMemorySaver()

call_transcript = ""
with open("call-reports.json", "r") as f:
    call_transcripts = json.load(f) 
    lines = call_transcripts[0]["transcript"]
    call_transcript = "\n".join(lines)


     
def summarize_call_transcript_node(state: AgentState):
     logger.info("node.start summarize_call_transcript_node")
     response  = base_llm.invoke(
        f"Summarize the following call transcript: {state['text']}")
     logger.info("node.end summarize_call_transcript_node")
     return { "messages": [response] , "summary" : response.content}

@tool("generate_action_items", args_schema=ActionItemRequest)
def generate_action_items(action_items: list[ActionItem]) -> ToolResponse:
    
    """This is a tool creates action items in a task management system. It takes a list of action items with description, owner and due date and creates those action items in the task management system."""


    logger.info("tool.start generate_action_items")
    created_tasks = []
    for item in action_items:
        task = ActionItemResult(
            description=item.description,
            owner=item.owner,
            due_date=item.due_date,
            status="created")
        created_tasks.append(task)
    
    tool_response = ToolResponse(   
        status="success",
        created_tasks=created_tasks,
        count=len(created_tasks)
    )

    if not action_items:
        logger.info("tool.end generate_action_items - no action items to create")
        return ToolResponse(
                status="success",
                created_tasks=[],
                count=0
            )
    logger.info("Created action items:")
    return tool_response



def pending_actions_node(state: AgentState):
     logger.info("node.start pending_actions_node")
     action_items_prompt = PromptTemplate.from_template(
         "Extract action items from this transcript. "
         "Call ONLY the tool `generate_action_items` with arguments in this exact schema: "
         "{{\"action_items\": [{{\"description\": \"...\", \"owner\": \"...\", \"due_date\": \"...\"}}]}}. "
         "Use the exact key `due_date` (snake_case). "
         "If no actions exist, pass {{\"action_items\": []}}. "
         "Transcript:\n{text}"
     )
     chain = action_items_prompt | llm
     response = chain.invoke({"text": state["text"]})
     logger.info("Model response: %s", response)
     logger.info("node.end pending_actions_node")
     return { "messages": [response] }


def local_gemma_model() -> ChatOpenAI:
      return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="google/gemma-3-4b",
        api_key="fake-api-key"
    )

base_llm = local_gemma_model()

agent_state = AgentState(text="", summary="", action_items=[], messages=[])
agent_state = agent_state | {"text": call_transcript}


tools = [generate_action_items]
tool_node = ToolNode(tools)

llm = base_llm.bind_tools(tools)

agent  = StateGraph(AgentState )

agent.add_node(summarize_call_transcript_node)
agent.add_node(pending_actions_node)
agent.add_node("tools",tool_node)


agent.add_edge(START, "summarize_call_transcript_node")
agent.add_edge("summarize_call_transcript_node", "pending_actions_node")
agent.add_conditional_edges("pending_actions_node", tools_condition)
agent.add_edge("tools", END)


app = agent.compile(checkpointer=store)

logger.info("agent.start")

config = {"configurable": {"thread_id": "call-report-001"}}
for mode, payload in app.stream(agent_state, config=config, stream_mode=["updates", "values"]):
    if mode == "updates":
        for node, output in payload.items():
            logger.info("node.end %s", node)
            logger.info("node.output %s", output)
          
            if "messages" in output:
                for msg in output["messages"]:
                    if isinstance(msg, ToolMessage):
                        logger.info("tool.end %s", msg.name)
                        logger.info("tool.result %s", msg.content)

    elif mode == "values":
        logger.info("state.snapshot %s", payload)


for cp in store.list(config={"configurable": {"thread_id": "call-report-001"}}):
    logger.info("Checkpoint: %s", cp)


logger.info("agent.end")
