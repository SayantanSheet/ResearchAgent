import io
from contextlib import redirect_stdout

import streamlit as st

# Reuse everything we already built: the agent loop, tools and memory.
from groq_agent import load_memory, run_agent, save_memory

st.set_page_config(page_title="Research Agent", page_icon="🔎")
st.title("🔎 Research Agent")
st.caption("Groq + web search + calculator + memory")

# Streamlit re-runs this whole file on every click or message.
# st.session_state keeps values between those re-runs (while the page is open).
if "history" not in st.session_state:
    st.session_state.history = load_memory()  # the agent's memory (from memory.json)
    st.session_state.chat = list(st.session_state.history)  # what is shown on screen

# ---- Sidebar ----
with st.sidebar:
    st.write(f"Remembering {len(st.session_state.history) // 2} earlier questions")
    if st.button("Clear memory"):
        st.session_state.history.clear()
        st.session_state.chat.clear()
        save_memory(st.session_state.history)
        st.rerun()


def show(text):
    # Streamlit treats "$...$" as a math formula, so "$5 and $10" would look broken.
    st.markdown((text or "").replace("$", "\\$"))


# ---- Show the conversation so far ----
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        show(msg["content"])

# ---- New question ----
question = st.chat_input("Ask something...")
if question:
    with st.chat_message("user"):
        show(question)

    with st.chat_message("assistant"):
        with st.status("Thinking...") as status:
            # run_agent prints "[using tool: ...]" lines. Catch them, to show them here.
            steps = io.StringIO()
            try:
                with redirect_stdout(steps):
                    answer = run_agent(question, st.session_state.history)
            except Exception as e:
                answer = f"Error: {e}"
            log = steps.getvalue().strip()
            if log:
                st.code(log, language=None)
                status.update(label="Steps the agent took", state="complete")
            else:
                status.update(label="Answered without tools", state="complete")
        show(answer)

    save_memory(st.session_state.history)
    st.session_state.chat.append({"role": "user", "content": question})
    st.session_state.chat.append({"role": "assistant", "content": answer})