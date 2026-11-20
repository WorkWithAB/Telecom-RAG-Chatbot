
import json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 1. File paths
BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_PATH = BASE_DIR / "data" / "embedded_chunks.npz"
CHUNKS_PATH = BASE_DIR / "data" / "unique_chunks.jsonl"

# 2. Load chunks
print("Loading chunks...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file]

documents = [chunk["Chunk"] for chunk in chunks]

# 3. Build BM25 index
print("Building BM25 index...")

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

# 4. Load embedding model
print("Loading embedding model...")

model = SentenceTransformer("intfloat/multilingual-e5-small")

# 5. Load vector embeddings
print("Loading vector embeddings...")

loaded = np.load(EMBEDDINGS_PATH)
print("Available arrays:", loaded.files)

embeddings = loaded[loaded.files[0]]

# 6. Hybrid search function
def hybrid_search(question, top_k=5, alpha=0.5):

    # BM25 search
    tokenized_query = question.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    # Vector search
    query_embedding = model.encode(
        "query: " + question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    vector_scores = np.dot(embeddings, query_embedding)

    # Normalize scores using min-max normalization
    def normalize(scores):
        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return np.zeros_like(scores)

        return (scores - minimum) / (maximum - minimum)

    normalized_bm25 = normalize(bm25_scores)
    normalized_vector = normalize(vector_scores)

    # Combine scores
    hybrid_scores = (
        alpha * normalized_vector
        + (1 - alpha) * normalized_bm25
    )

    # Retrieve top results
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    print("\n========== HYBRID SEARCH RESULTS ==========")

    for rank, index in enumerate(top_indices, start=1):

        chunk = chunks[index]

        print(f"\nRank: {rank}")
        print(f"Hybrid Score: {hybrid_scores[index]:.4f}")
        print(f"Vector Score: {vector_scores[index]:.4f}")
        print(f"BM25 Score: {bm25_scores[index]:.4f}")
        print(f"Source: {chunk.get('Source_Document')}")
        print(f"Chunk ID: {chunk.get('Chunk_ID')}")
        print(f"Text:\n{chunk.get('Chunk')}")
        print("-" * 70)


# 7. Run chatbot search
if __name__ == "__main__":

    print("\nTelecom Hybrid Search")

    while True:

        question = input("\nEnter your question (or type exit): ")

        if question.lower() == "exit":
            break

        hybrid_search(question, top_k=3)
