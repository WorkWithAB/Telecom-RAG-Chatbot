
from qdrant_client import QdrantClient, models

# Connect to local Qdrant
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "telecom_chunks"

print("Checking Qdrant collection...")

# Avoid creating the same collection twice
if client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists.")

else:
    print("Creating telecom_chunks collection...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

    print("Collection created successfully!")

# Verify collection
info = client.get_collection(COLLECTION_NAME)

print("\n========== COLLECTION INFO ==========")
print("Collection:", COLLECTION_NAME)
print("Vector dimensions:", info.config.params.vectors.size)
print("Points:", info.points_count)
print("Status:", info.status)
