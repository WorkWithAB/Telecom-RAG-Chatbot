
import numpy as np
import uuid

from pathlib import Path
from qdrant_client import QdrantClient, models

# Project paths
DATA_DIR = Path("data")
EMBEDDING_FILE = DATA_DIR / "embedded_chunks.npz"

# Qdrant configuration
COLLECTION_NAME = "telecom_chunks"
QDRANT_URL = "http://localhost:6333"

BATCH_SIZE = 128

print("Connecting to Qdrant...")

client = QdrantClient(url=QDRANT_URL)

# Verify collection exists
if not client.collection_exists(COLLECTION_NAME):
    raise RuntimeError(
        "Collection not found. Run create_qdrant_collection.py first."
    )

print("Loading existing embeddings...")

data = np.load(EMBEDDING_FILE, allow_pickle=False)

embeddings = data["embeddings"]
chunk_ids = data["chunk_ids"]
source_documents = data["source_documents"]
chunks = data["chunks"]

total = len(embeddings)

print("Total embeddings:", total)
print("Vector dimensions:", embeddings.shape[1])
print("Total chunks:", len(chunks))

# Validate data
assert embeddings.shape[1] == 384
assert len(embeddings) == len(chunk_ids) == len(chunks)
assert len(source_documents) == total

print("\nStarting upload to Qdrant...")

for start in range(0, total, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total)

    points = []

    for i in range(start, end):

        # Stable UUID based on original chunk ID
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_ids[i]))
        )

        point = models.PointStruct(
            id=point_id,

            vector=embeddings[i].tolist(),

            payload={
                "chunk_id": str(chunk_ids[i]),
                "source_document": str(source_documents[i]),
                "text": str(chunks[i])
            }
        )

        points.append(point)

    # Upload this batch
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    print(f"Uploaded: {end}/{total}")

print("\n========== UPLOAD COMPLETE ==========")

# Verify database
info = client.get_collection(COLLECTION_NAME)

print("Collection:", COLLECTION_NAME)
print("Expected vectors:", total)
print("Stored vectors:", info.points_count)
print("Collection status:", info.status)

if info.points_count == total:
    print("\nSUCCESS: All vectors stored in Qdrant!")
else:
    print("\nWARNING: Stored count does not match expected count.")
