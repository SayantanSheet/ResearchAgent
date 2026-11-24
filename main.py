import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()  # reads GEMINI_API_KEY from .env into the environment

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60_000),  # ms; fail instead of hanging forever
)

# Tried in order; if one is overloaded (503) or times out, fall back to the next.
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"]
RETRIES_PER_MODEL = 3


def generate(contents):
    last_error = None
    for model in MODELS:
        for attempt in range(RETRIES_PER_MODEL):
            try:
                return client.models.generate_content(model=model, contents=contents)
            except errors.ClientError as e:
                if e.code == 429:  # quota used up for this model, try the next one
                    last_error = e
                    break
                raise  # other 4xx = problem with our request
            except Exception as e:  # 503 overload or timeout
                last_error = e
                if attempt < RETRIES_PER_MODEL - 1:
                    wait = 2 ** attempt
                    print(f"{model} failed ({type(e).__name__}), retrying in {wait}s...")
                    time.sleep(wait)
        print(f"{model} unavailable, trying next model...")
    raise last_error


response = generate("What is agentic AI? Explain in two lines.")
print(response.text)


