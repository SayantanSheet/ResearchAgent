from groq_agent import AVAILABLE_FUNCTIONS, TOOLS

# Reuse the Researcher, the Writer and the agent loop we already built.
from multi_agent import RESEARCHER_PROMPT, WRITER_PROMPT, run_worker

# ---- 1. Shared workspace: where the team keeps its work ----
# The Researcher's notes are stored here, so the Manager doesn't have to
# copy long notes back and forth (that would waste tokens).
workspace = {"notes": "", "report": ""}


# ---- 2. The team members become the Manager's TOOLS (agents as tools) ----
def ask_researcher(task: str) -> str:
    print(f"\n[Manager -> Researcher] {task}")
    notes = run_worker(
        "Researcher",
        RESEARCHER_PROMPT,
        task,
        tool_names=["web_search", "get_current_time"],
        tool_limits={"web_search": 3},  # at most 3 searches per research round
    )
    workspace["notes"] += "\n" + notes  # keep earlier research too
    return notes  # the Manager reads the notes to judge them


def ask_writer(instructions: str) -> str:
    print(f"\n[Manager -> Writer] {instructions}")
    task = f"{instructions}\n\nResearch notes:\n{workspace['notes']}"
    if workspace["report"]:
        task += f"\n\nYour previous draft (improve it):\n{workspace['report']}"
    report = run_worker("Writer", WRITER_PROMPT, task)
    workspace["report"] = report
    return report  # the Manager reads the draft to check it


# ---- 3. The same 3 steps as every tool: function, description, name ----
TOOLS.extend(
    [
        {
            "type": "function",
            "function": {
                "name": "ask_researcher",
                "description": "Give a research task to the Researcher agent. It searches "
                "the web and returns bullet-point notes with sources. "
                "Call it again with a more specific task if the notes are thin or outdated.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "What to research, e.g. 'Latest agentic AI news in 2026'",
                        }
                    },
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_writer",
                "description": "Ask the Writer agent to write (or improve) the report, "
                "using all research notes collected so far. Returns the draft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instructions": {
                            "type": "string",
                            "description": "Topic and focus of the report, or what to fix "
                            "in the previous draft",
                        }
                    },
                    "required": ["instructions"],
                },
            },
        },
    ]
)
AVAILABLE_FUNCTIONS["ask_researcher"] = ask_researcher
AVAILABLE_FUNCTIONS["ask_writer"] = ask_writer


# ---- 4. The Manager's job: plan, delegate, check the work ----
MANAGER_PROMPT = (
    "You are the Manager of a small team: a Researcher and a Writer. "
    "You do not search or write reports yourself; you give work to your team with tools. "
    "If the request is simple (a greeting, or a question you can answer well without "
    "current information), answer it directly without the team. "
    "For a report: first use ask_researcher. Read the notes. If they are too few, "
    "off-topic or outdated, use ask_researcher again with a more specific task "
    "(at most 2 research rounds in total). "
    "Then use ask_writer. Read the draft. If something important is missing or wrong, "
    "use ask_writer once more with clear feedback. "
    "When the report is good, reply with a short 1-2 line note to the user. "
    "Do not repeat the whole report."
)

if __name__ == "__main__":
    request = input("What do you need? ").strip()
    if request:
        # The Manager is also just an agent: the same run_worker loop,
        # with its own prompt and its own tools (its team).
        reply = run_worker(
            "Manager",
            MANAGER_PROMPT,
            request,
            tool_names=["ask_researcher", "ask_writer"],
            max_steps=8,
        )
        if workspace["report"]:
            print("\n========== Final report ==========\n")
            print(workspace["report"])
            with open("report.md", "w", encoding="utf-8") as f:
                f.write(workspace["report"])
            print("\n(Report saved to report.md)")
        print(f"\n[Manager]: {reply}")