import os

import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

# Reuse the model caller we already built (with fallback between models).
from multi_agent import ask_model

DOCS_FOLDER = "docs"  # put your .txt, .md and .pdf files here
CHUNK_SIZE = 500  # characters per chunk
OVERLAP = 100  # characters shared between neighbouring chunks
TOP_K = 3  # how many chunks to give the model for each question

RAG_PROMPT = (
    "You answer questions using ONLY the context taken from the user's documents. "
    "Each piece of context starts with its source in [brackets]. "
    "After each fact, write its source, like [handbook.txt]. "
    "If the context does not contain the answer, say: "
    "'I couldn't find this in your documents.' Never use outside knowledge."
)


# ---- Step 1: LOAD - read every document in the folder ----
def load_documents(folder):
    docs = []  # list of (source, text)
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if filename.endswith((".txt", ".md")):
            with open(path, encoding="utf-8") as f:
                docs.append((filename, f.read()))
        elif filename.endswith(".pdf"):
            reader = PdfReader(path)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                docs.append((f"{filename} (page {page_number})", text))
    return docs


# ---- Step 2: CHUNK - cut long text into small overlapping pieces ----
def split_into_chunks(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap  # step forward, but keep some overlap
    return chunks


# ---- Step 3: EMBED - turn every chunk into a list of numbers (a vector) ----
print("Loading the embedding model (the first time it downloads ~70 MB)...")
embedder = TextEmbedding("BAAI/bge-small-en-v1.5")


def normalize(vectors):
    # Make every vector length 1, so a simple dot product = cosine similarity.
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def build_index(folder):
    sources, chunks = [], []
    for source, text in load_documents(folder):
        for chunk in split_into_chunks(text):
            sources.append(source)
            chunks.append(chunk)
    if not chunks:
        raise SystemExit(f"No text found. Put .txt, .md or .pdf files in the '{folder}' folder.")
    vectors = normalize(np.array(list(embedder.embed(chunks))))
    print(f"Indexed {len(chunks)} chunks from {len(set(sources))} sources.")
    return sources, chunks, vectors


# ---- Step 4: RETRIEVE - find the chunks whose meaning is closest to the question ----
def search(question, index, k=TOP_K):
    sources, chunks, vectors = index
    query_vector = normalize(np.array(list(embedder.query_embed(question))[0]))
    scores = vectors @ query_vector  # one similarity score per chunk (-1 to 1)
    best = np.argsort(scores)[::-1][:k]  # positions of the k highest scores
    return [(sources[i], chunks[i], float(scores[i])) for i in best]


# ---- Step 5: GENERATE - give the best chunks + the question to the model ----
def answer(question, index):
    results = search(question, index)
    print("\n[Retrieved chunks]")
    for source, chunk, score in results:
        print(f"  {score:.2f}  [{source}]  {chunk[:70]!r}...")

    context = "\n\n".join(f"[{source}]\n{chunk}" for source, chunk, _ in results)
    messages = [
        {"role": "system", "content": RAG_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return ask_model(messages, tools=[]).choices[0].message.content


if __name__ == "__main__":
    index = build_index(DOCS_FOLDER)
    print("Ask about your documents. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if question:
            print("\nAnswer:", answer(question, index))