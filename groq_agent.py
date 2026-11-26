import datetime
import json
import os
import re

from dotenv import load_dotenv
from groq import APIStatusError, Groq

load_dotenv()  # reads GROQ_API_KEY from the .env file
client = Groq()

# Free-plan models on Groq. If one fails, try the next.
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

# Models that can use Groq's built-in browser_search, tried in this order.
SEARCH_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

# Memory: how many past messages the agent remembers (10 = last 5 questions + answers).
# Kept small because the free plan allows only ~8,000 tokens per minute.
MAX_HISTORY = 10

# Long-term memory: the conversation is saved in this file, so it survives a restart.
MEMORY_FILE = "memory.json"

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Use web_search for news, current events, prices, or any fact you are not sure about. "
    "Use get_current_time when you need today's date or time. "
    "Only state facts you got from tools or from earlier in this conversation, "
    "and list the source links at the end. "
    "Use the earlier conversation to understand follow-up questions like 'tell me more about it'. "
    "If you don't have the answer, say you don't know. Never guess."
)


# ---- 1. Tools: normal Python functions ----
def get_current_time() -> str:
    return datetime.datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    # eval() can run any Python code, so only allow numbers and math symbols.
    # "**" (power) is blocked too, because something like 9**9**9 would freeze the laptop.
    allowed = "0123456789+-*/(). "
    if not all(ch in allowed for ch in expression) or "**" in expression:
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def clean_search_answer(message):
    # gpt-oss adds citation marks like 【2†L4-L11】 that mean nothing to us. Remove them.
    text = re.sub(r"【[^】]*】", "", message.content or "").strip()

    # Groq also sends what the search tool really did, in message.executed_tools:
    #  - "browser.search": the list of search results (all hits, read or not)
    #  - "browser.open":   a page the model actually opened and read
    titles = {}  # url -> title, from the search results
    found = []  # every search hit
    opened = []  # pages that were really opened and read
    for tool in getattr(message, "executed_tools", None) or []:
        if tool.search_results and tool.search_results.results:
            for r in tool.search_results.results:
                if r.url:
                    found.append(r.url)
                    titles[r.url] = r.title or r.url
        if "open" in tool.type:
            match = re.search(r"URL: (\S+)", tool.output or "")
            if match:
                opened.append(match.group(1))

    # Best sources = the pages it read. If it read none, use the top 3 search hits.
    sources = list(dict.fromkeys(opened or found[:3]))  # dict.fromkeys removes duplicates
    if sources:
        text += "\n\nSources (use ONLY these links, never invent others):\n"
        text += "\n".join(f"- {titles.get(url, url)}: {url}" for url in sources[:5])
    return text or "No results found."


def web_search(query: str) -> str:
    # The search runs on Groq's servers (its built-in "browser_search" tool),
    # so our own slow network doesn't matter. Our tool asks a second model
    # to do the searching: one agent using another agent as its tool.
    # Each model has its own daily token limit, so if one is used up, try the next.
    for model in SEARCH_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Search the web for: {query}\n"
                        "Reply with the key facts as short bullet points. "
                        "Put the source URL after each fact.",
                    }
                ],
                tools=[{"type": "browser_search"}],
                tool_choice="required",
                reasoning_effort="low",  # Groq recommends low, so searches stay short
            )
            message = response.choices[0].message
            return clean_search_answer(message)
        except APIStatusError as e:
            if e.status_code in (429, 503):
                print(f"[search: {model} limit reached, trying next model...]")
                continue
            print(f"[search error: {e}]")
            return f"Search failed: {e}"
        except Exception as e:
            print(f"[search error: {e}]")
            return f"Search failed: {e}"
    # All search models are used up. Tell the main model clearly, so it
    # doesn't keep trying again and again.
    print("[search: all search models have reached their limit]")
    return (
        "Search is not available right now (daily limit reached). "
        "Do not call web_search again. Tell the user to try again later."
    )


# ---- 2. Describe the tools, so the model knows what each does and when to use it ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current day of the week, date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web and returns key facts with their "
            "source URLs. Use it for news and up-to-date facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short search query, e.g. 'latest news agentic AI'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculates a math expression, e.g. '25 * 4 + 10'. "
            "Use it for any arithmetic instead of doing math yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression with numbers and + - * / ( ) only",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# Tool name -> the real Python function to run.
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "web_search": web_search,
    "calculate": calculate,
}


def run_tool(name, arguments):
    func = AVAILABLE_FUNCTIONS.get(name)
    if func is None:
        return f"Error: there is no tool called {name}."
    try:
        args = json.loads(arguments or "{}")
        return func(**args)
    except Exception as e:
        # Send the error back to the model, so it can try again differently.
        return f"Error while running {name}: {e}"


def call_model(messages):
    for model in MODELS:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except APIStatusError as e:
            busy = e.status_code in (429, 503)
            bad_tool_call = e.status_code == 400 and "tool_use_failed" in str(e)
            if busy or bad_tool_call:
                print(f"{model} failed ({e.status_code}), trying next model...")
                continue
            raise  # any other error = problem with our request
    raise RuntimeError("All models are busy or out of quota. Try again later.")


# ---- 3. The agent loop: think -> use a tool -> look at the result -> answer ----
def run_agent(question, history, max_steps=6):
    # Rules first, then the memory (earlier questions + answers), then the new question.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": question})

    for _ in range(max_steps):
        message = call_model(messages).choices[0].message

        # No tool requested -> this is the final answer.
        if not message.tool_calls:
            answer = message.content
            # Save only the question and the final answer to memory.
            # The tool calls and search results are left out, so memory stays small.
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            del history[:-MAX_HISTORY]  # forget the oldest messages beyond the limit
            return answer

        # Remember that the model asked for these tools.
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        # Run each tool and send its result back to the model.
        for tc in message.tool_calls:
            print(f"[using tool: {tc.function.name} {tc.function.arguments}]")
            result = run_tool(tc.function.name, tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Stopped: too many steps without a final answer."


# ---- 4. Long-term memory: save to / load from a JSON file ----
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []  # first run: nothing to remember yet
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)  # JSON text in the file -> Python list
    except (json.JSONDecodeError, OSError):
        print("(memory file is broken, starting fresh)")
        return []


def save_memory(history):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        # Python list -> JSON text in the file
        json.dump(history, f, ensure_ascii=False, indent=2)


# ---- 5. Chat with the agent (with memory) ----
if __name__ == "__main__":
    history = load_memory()  # read old conversation from the file
    print(f"(remembering {len(history) // 2} earlier questions)")
    print("Ask a question. Type 'reset' to clear memory, 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "reset":
            history.clear()
            save_memory(history)  # empty the file too
            print("(memory cleared)")
            continue
        if question:
            print("\nAgent:", run_agent(question, history))
            save_memory(history)  # save after every answer