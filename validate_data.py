
import json
import numpy as np
from pathlib import Path
from collections import Counter

DATA_PATH = Path("data/unique_chunks.jsonl")
EMBEDDINGS_PATH = Path("data/embedded_chunks.npz")

print("\n========== DATA VALIDATION ==========\n")

# 1. Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file if line.strip()]

print("Total dataset chunks:", len(chunks))

# 2. Check empty chunks
empty_chunks = [
    i for i, chunk in enumerate(chunks)
    if not chunk.get("Chunk", "").strip()
]

print("Empty chunks:", len(empty_chunks))

# 3. Check duplicate chunk IDs
chunk_ids = [chunk.get("Chunk_ID") for chunk in chunks]
id_counts = Counter(chunk_ids)

duplicate_ids = {
    key: count for key, count in id_counts.items()
    if key is not None and count > 1
}

print("Duplicate Chunk IDs:", len(duplicate_ids))

# 4. Load embeddings and metadata
data = np.load(EMBEDDINGS_PATH)

embeddings = data["embeddings"]
embedded_ids = data["chunk_ids"]
embedded_chunks = data["chunks"]

print("\nEmbedding shape:", embeddings.shape)
print("Embedded chunk count:", len(embedded_chunks))
print("Embedded ID count:", len(embedded_ids))

# 5. Check alignment
dataset_ids = [str(x) for x in chunk_ids]
embedded_ids = [str(x) for x in embedded_ids]

print("\nDataset and embedding counts match:",
      len(chunks) == len(embeddings))

print("Chunk IDs match in same order:",
      dataset_ids == embedded_ids)

# 6. Check invalid embeddings
print("Embeddings contain NaN:",
      bool(np.isnan(embeddings).any()))

print("Embeddings contain infinity:",
      bool(np.isinf(embeddings).any()))

print("\n========== VALIDATION COMPLETE ==========")
