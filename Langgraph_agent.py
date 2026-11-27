from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Reuse our own tool functions (this also loads GROQ_API_KEY from .env).
import groq_agent

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Use web_search for news, current events, or any fact you are not sure about. "
    "Use get_current_time when you need today's date or time. "
    "Use calculate for any arithmetic. "
    "Only state facts you got from tools or from earlier in this conversation, "
    "and list the source links at the end. Never invent links. "
    "If you don't have the answer, say you don't know."
)


# ---- 1. Tools: @tool reads the type hints and the docstring, ----
# ---- so we don't write the JSON description by hand anymore.  ----
@tool
def get_current_time() -> str:
    """Returns the current day of the week, date and time."""
    return groq_agent.get_current_time()


@tool
def calculate(expression: str) -> str:
    """Calculates a math expression with numbers and + - * / ( ), e.g. '25 * 4 + 10'."""
    return groq_agent.calculate(expression)


@tool
def web_search(query: str) -> str:
    """Searches the web and returns key facts with source URLs. Use it for news and up-to-date facts."""
    return groq_agent.web_search(query)


TOOLS = [get_current_time, calculate, web_search]


# ---- 2. The model, with fallbacks (this replaces our call_model loop) ----
models = [ChatGroq(model=name, max_retries=1).bind_tools(TOOLS) for name in groq_agent.MODELS]
llm = models[0].with_fallbacks(models[1:])  # if one model fails, try the next


# ---- 3. The "agent" node: think and decide ----
def agent_node(state: MessagesState):
    # Memory: keep only the last 12 messages, starting at a user question
    # (same idea as MAX_HISTORY, so we don't run out of tokens).
    recent = trim_messages(
        state["messages"], strategy="last", token_counter=len, max_tokens=12, start_on="human"
    )
    reply = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + recent)
    return {"messages": [reply]}  # LangGraph adds this to the conversation


# ---- 4. Build the graph: this IS our old agent loop, drawn as boxes and arrows ----
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)  # box 1: the model thinks
graph.add_node("tools", ToolNode(TOOLS))  # box 2: run the tools it asked for
graph.add_edge(START, "agent")  # start -> agent
graph.add_conditional_edges("agent", tools_condition)  # tool asked? -> tools, else -> end
graph.add_edge("tools", "agent")  # tools -> back to agent
app = graph.compile(checkpointer=InMemorySaver())  # checkpointer = memory between questions


if __name__ == "__main__":
    # thread_id = one conversation. recursion_limit = like our max_steps.
    config = {"configurable": {"thread_id": "chat-1"}, "recursion_limit": 12}
    print("Ask anything. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        # stream = see each step (each box the graph visits) as it happens
        for step in app.stream({"messages": [("user", question)]}, config, stream_mode="updates"):
            for node, update in step.items():
                for msg in update["messages"]:
                    if node == "agent" and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  [agent -> tools: {tc['name']} {tc['args']}]")
                    elif node == "agent":
                        print("\nAgent:", msg.content)