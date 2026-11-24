import datetime
import json

from dotenv import load_dotenv
from groq import APIStatusError, Groq

load_dotenv()  # reads GROQ_API_KEY from the .env file
client = Groq()

# Free-plan models on Groq. If one fails, try the next.
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Use web_search for news, current events, prices, or any fact you are not sure about. "
    "Use get_current_time when you need today's date or time. "
    "Only state facts you got from tools, and list the source links at the end. "
    "If the tools don't give you the answer, say you don't know. Never guess."
)


# ---- 1. Tools: normal Python functions ----
def get_current_time() -> str:
    return datetime.datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


def web_search(query: str) -> str:
    # The search runs on Groq's servers (its built-in "browser_search" tool),
    # so our own slow network doesn't matter. Our tool asks a second, smaller
    # model to do the searching: one agent using another agent as its tool.
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
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
        return response.choices[0].message.content or "No results found."
    except Exception as e:
        print(f"[search error: {e}]")
        return f"Search failed: {e}"


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
]

# Tool name -> the real Python function to run.
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "web_search": web_search,
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
def run_agent(question, max_steps=6):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        message = call_model(messages).choices[0].message

        # No tool requested -> this is the final answer.
        if not message.tool_calls:
            return message.content

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


# ---- 4. Chat with the agent: type a question, or 'exit' to quit ----
if __name__ == "__main__":
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if question:
            print("\nAgent:", run_agent(question))