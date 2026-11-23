
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Configuration
MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "telecom_chunks"

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Connecting to Qdrant...")

client = QdrantClient(
    url="http://localhost:6333"
)

# Enter your question
question = input("\nEnter your telecom question: ")

# Generate query embedding
query_embedding = model.encode(
    "query: " + question,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\nSearching Qdrant...")

# Search database
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding.tolist(),
    limit=5,
    with_payload=True
).points

# Display results
print("\n========== QDRANT SEMANTIC SEARCH ==========")

for rank, result in enumerate(results, start=1):

    payload = result.payload

    print(f"\nRank: {rank}")
    print(f"Similarity Score: {result.score:.4f}")
    print(f"Source: {payload['source_document']}")
    print(f"Chunk ID: {payload['chunk_id']}")

    print("\nText:")
    print(payload['text'][:700])
    print("-" * 60)

print("\nSearch completed!")
