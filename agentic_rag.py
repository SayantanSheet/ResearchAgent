from groq_agent import AVAILABLE_FUNCTIONS, TOOLS
from multi_agent import run_worker

# Reuse our RAG pieces: build the index, and search it.
from rag_agent import DOCS_FOLDER, build_index, search

index = None  # filled in when the program starts


# ---- 1. RAG becomes a TOOL (same 3 steps as every tool) ----
def search_documents(query: str) -> str:
    results = search(query, index)
    # Give the agent each chunk with its source and score, so it can judge them.
    return "\n\n".join(
        f"[{source}] (score {score:.2f})\n{chunk}" for source, chunk, score in results
    )


TOOLS.append(
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Searches the user's own files (handbook, notes, PDFs) and returns "
            "the most relevant pieces with their source file and page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, e.g. 'paid leave days per year'",
                    }
                },
                "required": ["query"],
            },
        },
    }
)
AVAILABLE_FUNCTIONS["search_documents"] = search_documents


# ---- 2. The rule: documents first, internet only if needed ----
AGENT_PROMPT = (
    "You are a research assistant with two sources of information. "
    "1) search_documents: the user's own files. ALWAYS try this first. "
    "2) web_search: the internet. Use it ONLY if the documents do not answer the question, "
    "or if the question needs current news. "
    "Read the document results carefully: if they do not really answer the question, "
    "then use web_search. "
    "Use get_current_time if you need today's date. "
    "In your answer, say clearly where each fact came from: "
    "(from your documents: [file name]) or (from the web: URL). "
    "If neither source has the answer, say you don't know. Never guess."
)

if __name__ == "__main__":
    index = build_index(DOCS_FOLDER)
    print("Ask anything. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if question:
            answer = run_worker(
                "Agent",
                AGENT_PROMPT,
                question,
                tool_names=["search_documents", "web_search", "get_current_time"],
                tool_limits={"search_documents": 3, "web_search": 2},
            )
            print("\nAnswer:", answer)