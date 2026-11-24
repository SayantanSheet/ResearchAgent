import time
import urllib.request

from ddgs import DDGS

# 1. Can Python reach these websites at all?
print("--- Internet check ---")
for url in ["https://duckduckgo.com", "https://en.wikipedia.org", "https://search.brave.com"]:
    start = time.time()
    try:
        urllib.request.urlopen(url, timeout=15)
        print(f"OK    {url}  ({time.time() - start:.1f}s)")
    except Exception as e:
        print(f"FAIL  {url}  ({type(e).__name__}: {e})")

# 2. Try each search engine one by one, with a longer timeout (default is only 5s).
print("\n--- Search engines ---")
for backend in ["duckduckgo", "wikipedia", "brave", "mojeek", "yahoo", "startpage"]:
    start = time.time()
    try:
        results = DDGS(timeout=15).text("artificial intelligence", max_results=3, backend=backend)
        print(f"OK    {backend:<11} {len(results)} results  ({time.time() - start:.1f}s)")
    except Exception as e:
        print(f"FAIL  {backend:<11} {type(e).__name__}  ({time.time() - start:.1f}s)")