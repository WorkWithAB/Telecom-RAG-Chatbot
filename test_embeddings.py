
import numpy as np

data = np.load("data/embedded_chunks.npz")

embeddings = data["embeddings"]
chunk_ids = data["chunk_ids"]
chunks = data["chunks"]

print("Embeddings shape:", embeddings.shape)

# Select first chunk
query_vector = embeddings[0]

# Calculate cosine similarity
similarities = embeddings @ query_vector

# Get top 5 similar chunks
top_indices = np.argsort(similarities)[::-1][:5]

print("\nTop 5 Similar Chunks:\n")

for index in top_indices:

    print("Chunk ID:", chunk_ids[index])
    print("Similarity:", round(float(similarities[index]), 4))
    print("Text:", chunks[index][:300])
    print("-" * 60)
