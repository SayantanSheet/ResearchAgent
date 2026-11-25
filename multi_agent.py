from groq import APIStatusError

# Reuse what we already built: the Groq client, models, tools and run_tool.
from groq_agent import MODELS, TOOLS, client, run_tool

# ---- 1. Each agent = its own system prompt (its "job") + its own tools ----
RESEARCHER_PROMPT = (
    "You are a Researcher. Your only job is to collect facts. "
    "Use web_search to find information about the topic. Do at most 3 searches. "
    "First call get_current_time, and include the current year in your search queries. "
    "Prefer the newest sources. "
    "Reply ONLY with short bullet-point notes. Put the source URL after each fact. "
    "Do not write an essay. If the search fails, say clearly that you found nothing."
)

WRITER_PROMPT = (
    "You are a Writer. You get a topic and research notes from a Researcher. "
    "Write a clear, easy-to-read report in Markdown with these parts: "
    "a title, a 2-3 line summary, the key points, and a 'Sources' list with the URLs. "
    "Use ONLY facts from the notes. Do not add anything new. "
    "If the notes are empty or say the search failed, "
    "say that there was not enough information to write the report."
)


# ---- 2. Call the model (same as call_model, but each agent can have different tools) ----
def ask_model(messages, tools):
    for model in MODELS:
        try:
            kwargs = {"model": model, "messages": messages}
            if tools:  # the Writer has no tools, so we don't send any
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)
        except APIStatusError as e:
            busy = e.status_code in (429, 503)
            bad_tool_call = e.status_code == 400 and "tool_use_failed" in str(e)
            if busy or bad_tool_call:
                print(f"  {model} failed ({e.status_code}), trying next model...")
                continue
            raise
    raise RuntimeError("All models are busy or out of quota. Try again later.")


# ---- 3. One loop for every agent: the same think -> tool -> answer loop as before ----
def run_worker(name, system_prompt, task, tool_names=(), max_steps=6, tool_limits=None):
    # Give this agent only its own tools, not all of them.
    tools = [t for t in TOOLS if t["function"]["name"] in tool_names]
    # tool_limits = the most times a tool may be used, e.g. {"web_search": 3}.
    # The prompt only ASKS the model to stop; this code makes sure it does.
    tool_limits = tool_limits or {}
    used = {}  # how many times each tool has been used so far
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    for _ in range(max_steps):
        message = ask_model(messages, tools).choices[0].message

        if not message.tool_calls:
            return message.content or ""

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

        for tc in message.tool_calls:
            tool = tc.function.name
            limit = tool_limits.get(tool)  # None = no limit for this tool
            print(f"  [{name} using tool: {tool} {tc.function.arguments}]")
            if tool not in tool_names:
                result = f"Error: you are not allowed to use {tool}."
            elif limit is not None and used.get(tool, 0) >= limit:
                print(f"  [{name}: {tool} limit of {limit} reached, blocked]")
                result = (
                    f"Limit reached: you already used {tool} {limit} times. "
                    "Do not call it again. Finish your answer with what you have."
                )
            else:
                used[tool] = used.get(tool, 0) + 1
                result = run_tool(tool, tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return f"{name} stopped: too many steps without a final answer."


# ---- 4. The two agents ----
def researcher(topic):
    return run_worker(
        "Researcher",
        RESEARCHER_PROMPT,
        f"Research this topic: {topic}",
        tool_names=["web_search", "get_current_time"],
        tool_limits={"web_search": 3},  # at most 3 searches, enforced by code
    )


def writer(topic, notes):
    return run_worker(
        "Writer",
        WRITER_PROMPT,
        f"Topic: {topic}\n\nResearch notes:\n{notes}",
    )  # no tool_names: the Writer only writes


# ---- 5. The Manager: decides who works, in what order, and passes the work along ----
def manager(topic):
    print("\n[Manager] Step 1: sending the topic to the Researcher...")
    notes = researcher(topic)
    print("\n----- Researcher's notes -----")
    print(notes)

    print("\n[Manager] Step 2: sending the notes to the Writer...")
    report = writer(topic, notes)
    return report


if __name__ == "__main__":
    topic = input("Topic for the report: ").strip()
    if topic:
        report = manager(topic)
        print("\n========== Final report ==========\n")
        print(report)
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n(Report saved to report.md)")