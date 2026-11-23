
import numpy as np
from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "telecom_chunks"

# Load one existing vector as a test query
data = np.load("data/embedded_chunks.npz", allow_pickle=False)

query_vector = data["embeddings"][0].tolist()

print("Searching Qdrant...")

# Perform vector similarity search
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5,
    with_payload=True
).points

print("\n========== QDRANT SEARCH RESULTS ==========")

for rank, result in enumerate(results, start=1):

    print(f"\nRank: {rank}")
    print(f"Similarity Score: {result.score:.4f}")
    print(f"Chunk ID: {result.payload['chunk_id']}")
    print(f"Source: {result.payload['source_document']}")

    text = result.payload["text"]

    print("Text:", text[:500], "...")
    
print("\nQdrant vector search completed!")
