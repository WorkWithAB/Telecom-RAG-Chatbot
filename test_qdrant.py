
from qdrant_client import QdrantClient

print("Connecting to Qdrant...")

client = QdrantClient(url="http://localhost:6333")

# Check Qdrant server
collections = client.get_collections()

print("\nConnection successful!")
print("Available collections:", collections.collections)
