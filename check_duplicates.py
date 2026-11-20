
import numpy as np
from collections import Counter

data = np.load("data/embedded_chunks.npz")

embeddings = data["embeddings"]
chunk_ids = data["chunk_ids"].astype(str)
chunks = data["chunks"].astype(str)

print("========== DATASET VALIDATION ==========")

print("Total embeddings:", len(embeddings))
print("Unique Chunk IDs:", len(set(chunk_ids)))
print("Unique text chunks:", len(set(chunks)))

# Find duplicate IDs
counts = Counter(chunk_ids)

duplicates = {
    chunk_id: count
    for chunk_id, count in counts.items()
    if count > 1
}

print("\nDuplicate Chunk IDs:", len(duplicates))

for chunk_id, count in list(duplicates.items())[:10]:
    print(f"\nID: {chunk_id}")
    print("Occurrences:", count)

    indices = np.where(chunk_ids == chunk_id)[0]

    for index in indices:
        print("Index:", index)
        print("Text:", chunks[index][:150])

print("\n========== VALIDATION COMPLETE ==========")
