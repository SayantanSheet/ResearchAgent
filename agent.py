import datetime

from google import genai
from google.genai import errors, types

# The API key is read automatically from the GEMINI_API_KEY environment variable.
client = genai.Client()

# Free-tier quota is per model per day, so if one model is out of quota (429)
# or overloaded (503), try the next one.
MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"]


def get_current_time() -> str:
    """Returns the current day of the week, date and time."""
    return datetime.datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


config = types.GenerateContentConfig(tools=[get_current_time])


def ask(question):
    for model in MODELS:
        # A chat session is the recommended way to use automatic function calling.
        chat = client.chats.create(model=model, config=config)
        try:
            return chat.send_message(question)
        except errors.APIError as e:
            if e.code in (429, 503):
                print(f"{model} unavailable ({e.code}), trying next model...")
                continue
            raise  # any other error = problem with our request
    raise RuntimeError("All models are busy or out of quota. Try again later.")


response = ask("What time is it right now?")
print(response.text)