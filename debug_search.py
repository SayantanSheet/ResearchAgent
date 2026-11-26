# One-time check: what does Groq's browser_search really send back?
from groq_agent import client

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Search the web for: what is an MCP server"}],
    tools=[{"type": "browser_search"}],
    tool_choice="required",
    reasoning_effort="low",
)
message = response.choices[0].message

print("===== CONTENT (first 1500 characters) =====")
print((message.content or "")[:1500])

print("\n===== EXECUTED_TOOLS =====")
tools = message.executed_tools or []
print(f"{len(tools)} executed tool(s)")
for tool in tools:
    print(tool.model_dump_json(indent=2, exclude_none=True)[:3000])