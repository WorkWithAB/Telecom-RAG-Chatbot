
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Force CPU execution
device = "cpu"

# 2. Load multilingual embedding model
model_name = "intfloat/multilingual-e5-small"

print("Loading embedding model...")

model = SentenceTransformer(
    model_name,
    device=device
)

# 3. Load prepared chunks
input_file = "data/unique_chunks.jsonl"

records = []

with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        records.append(json.loads(line))

print("Total chunks:", len(records))

# 4. Extract text
texts = [
    "passage: " + record["Chunk"]
    for record in records
]

# 5. Generate embeddings in small batches
print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# 6. Save embeddings and metadata
output_file = "data/embedded_chunks.npz"

np.savez_compressed(
    output_file,
    embeddings=embeddings,
    chunk_ids=np.array([r["Chunk_ID"] for r in records]),
    source_documents=np.array(
        [r["Source_Document"] for r in records]
    ),
    chunks=np.array([r["Chunk"] for r in records])
)

print("\n========== EMBEDDING COMPLETE ==========")
print("Total embeddings:", len(embeddings))
print("Embedding dimensions:", embeddings.shape[1])
print("Saved to:", output_file)
