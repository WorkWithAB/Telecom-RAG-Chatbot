
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. DEFINE FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_PATH = BASE_DIR / "data" / "embedded_chunks.npz"
CHUNKS_PATH = BASE_DIR / "data" / "unique_chunks.jsonl"


# ==========================================
# 2. LOAD MULTILINGUAL EMBEDDING MODEL
# ==========================================

print("Loading multilingual embedding model...")

model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)


# ==========================================
# 3. LOAD STORED EMBEDDINGS AND METADATA
# ==========================================

print("\nLoading stored embeddings...")

loaded = np.load(EMBEDDINGS_PATH)

print("Available arrays:", loaded.files)

embeddings = loaded["embeddings"]

stored_chunk_ids = loaded["chunk_ids"]
stored_sources = loaded["source_documents"]
stored_chunks = loaded["chunks"]

print("Embedding shape:", embeddings.shape)


# ==========================================
# 4. LOAD ORIGINAL TEXT CHUNKS
# ==========================================

print("\nLoading text chunks...")

chunks = []

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:

    for line in file:
        chunks.append(json.loads(line))

print("Total chunks:", len(chunks))


# ==========================================
# 5. VALIDATE METADATA ALIGNMENT
# ==========================================

print("\n========== METADATA VALIDATION ==========")

print("Embedding count:", len(embeddings))
print("Stored chunk IDs:", len(stored_chunk_ids))
print("Stored sources:", len(stored_sources))
print("Stored text chunks:", len(stored_chunks))

counts_match = (
    len(embeddings)
    == len(stored_chunk_ids)
    == len(stored_sources)
    == len(stored_chunks)
    == len(chunks)
)

print("Embedding and metadata counts match:", counts_match)

ids_match = (
    len(stored_chunk_ids) == len(chunks)
    and all(
        str(stored_chunk_ids[i]) == str(chunks[i]["Chunk_ID"])
        for i in range(len(chunks))
    )
)

print("Chunk IDs match JSONL:", ids_match)

if not counts_match or not ids_match:
    raise ValueError(
        "Data validation failed! Check your embeddings and dataset."
    )

print("Data validation successful!")


# ==========================================
# 6. SEMANTIC SEARCH FUNCTION
# ==========================================

def semantic_search(question, top_k=5):

    print("\nSearching for relevant telecom information...")

    # Convert user question into an embedding
    query_embedding = model.encode(
        "query: " + question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Calculate similarity scores
    scores = np.dot(embeddings, query_embedding)

    # Retrieve more candidates for better diversity
    candidate_k = min(top_k * 10, len(scores))

    candidate_indices = np.argsort(scores)[::-1][:candidate_k]

    selected_indices = []

    seen_chunks = set()

    source_counts = {}

    # Maximum two chunks from one document
    max_per_source = 2

    # Select unique and diverse results
    for index in candidate_indices:

        chunk = chunks[index]

        text = chunk.get("Chunk", "").strip()

        source = chunk.get("Source_Document", "Unknown")

        # Skip empty or duplicate text
        if not text or text in seen_chunks:
            continue

        # Limit results from the same source
        if source_counts.get(source, 0) >= max_per_source:
            continue

        selected_indices.append(index)

        seen_chunks.add(text)

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(selected_indices) == top_k:
            break

    # Display search results
    print("\n========== SEARCH RESULTS ==========\n")

    if not selected_indices:
        print("No relevant results found.")
        return

    for rank, index in enumerate(selected_indices, start=1):

        chunk = chunks[index]

        print(f"Rank: {rank}")

        print(f"Similarity: {scores[index]:.4f}")

        print(
            f"Source: {chunk.get('Source_Document', 'Unknown')}"
        )

        print(
            f"Chunk ID: {chunk.get('Chunk_ID', 'Unknown')}"
        )

        print(f"Text:\n{chunk.get('Chunk', '')}")

        print("-" * 70)

    print("Total unique results:", len(selected_indices))


# ==========================================
# 7. TEST SEMANTIC SEARCH
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nEnter your telecom question: "
    )

    semantic_search(question, top_k=5)
