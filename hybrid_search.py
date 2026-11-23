
import json
import uuid
import numpy as np

from pathlib import Path
from collections import Counter

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_PATH = BASE_DIR / "data" / "unique_chunks.jsonl"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "telecom_chunks"

MODEL_NAME = "intfloat/multilingual-e5-small"

TOP_K = 3
CANDIDATE_K = 20

ALPHA = 0.5
RRF_K = 60

MMR_LAMBDA = 0.7

# =====================================================
# 2. LOAD CHUNKS
# =====================================================

print("Loading chunks...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

documents = [chunk["Chunk"] for chunk in chunks]

chunk_lookup = {
    str(chunk["Chunk_ID"]): chunk
    for chunk in chunks
}

print("Total chunks:", len(chunks))

# =====================================================
# 3. VALIDATE CHUNK IDS
# =====================================================

chunk_ids = [
    str(chunk["Chunk_ID"])
    for chunk in chunks
]

counts = Counter(chunk_ids)

duplicates = [
    chunk_id
    for chunk_id, count in counts.items()
    if count > 1
]

assert not duplicates, "Duplicate Chunk IDs detected!"

print("Chunk ID validation passed!")

# =====================================================
# 4. BUILD BM25 INDEX
# =====================================================

print("\nBuilding BM25 index...")

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

print("BM25 index created!")

# =====================================================
# 5. LOAD EMBEDDING MODEL
# =====================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Embedding model loaded!")

# =====================================================
# 6. CONNECT TO QDRANT
# =====================================================

print("\nConnecting to Qdrant...")

client = QdrantClient(url=QDRANT_URL)

assert client.collection_exists(
    collection_name=COLLECTION_NAME
), "Qdrant collection does not exist!"

collection_info = client.get_collection(COLLECTION_NAME)

assert collection_info.points_count == len(chunks), (
    "Qdrant point count does not match dataset!"
)

assert collection_info.config.params.vectors.size == 384, (
    "Qdrant vector dimensions mismatch!"
)

print("Qdrant connected!")
print("Stored points:", collection_info.points_count)

# =====================================================
# 7. MMR SEARCH
# =====================================================

def mmr_search(
    candidate_ids,
    relevance_scores,
    candidate_vectors,
    top_k=3,
    lambda_param=0.7
):

    if not candidate_ids:
        return []

    vectors = np.asarray(
        candidate_vectors,
        dtype=np.float32
    )

    relevance_scores = np.asarray(
        relevance_scores,
        dtype=np.float32
    )

    if len(candidate_ids) != len(vectors):
        raise ValueError("Candidate IDs and vectors count mismatch!")

    if len(candidate_ids) != len(relevance_scores):
        raise ValueError("Candidate IDs and relevance scores count mismatch!")

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    vectors = vectors / np.maximum(norms, 1e-12)

    # Calculate pairwise similarity
    similarity_matrix = np.dot(
        vectors,
        vectors.T
    )

    # Select the most relevant candidate first
    selected = [
        int(np.argmax(relevance_scores))
    ]

    while len(selected) < min(top_k, len(candidate_ids)):

        remaining = [
            i
            for i in range(len(candidate_ids))
            if i not in selected
        ]

        mmr_scores = []

        for position in remaining:

            relevance = relevance_scores[position]

            diversity_penalty = np.max(
                similarity_matrix[position, selected]
            )

            score = (
                lambda_param * relevance
                - (1 - lambda_param) * diversity_penalty
            )

            mmr_scores.append(score)

        best_position = remaining[
            int(np.argmax(mmr_scores))
        ]

        selected.append(best_position)

    return [
        candidate_ids[i]
        for i in selected
    ]

# =====================================================
# 8. HYBRID SEARCH
# =====================================================

def hybrid_search(
    question,
    top_k=TOP_K,
    alpha=ALPHA
):

    if not question.strip():
        print("Please enter a valid question.")
        return

    # -------------------------------------------------
    # A. BM25 SEARCH
    # -------------------------------------------------

    tokenized_query = question.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_indices = np.argsort(
        bm25_scores
    )[::-1][:CANDIDATE_K]

    bm25_candidates = {}

    for index in bm25_top_indices:

        chunk_id = str(
            chunks[index]["Chunk_ID"]
        )

        bm25_candidates[chunk_id] = float(
            bm25_scores[index]
        )

    # -------------------------------------------------
    # B. QUERY EMBEDDING
    # -------------------------------------------------

    query_embedding = model.encode(
        "query: " + question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_vector = query_embedding.astype(
        np.float32
    ).tolist()

    # -------------------------------------------------
    # C. QDRANT VECTOR SEARCH
    # -------------------------------------------------

    vector_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=CANDIDATE_K,
        with_payload=True,
        with_vectors=True
    ).points

    vector_candidates = {}

    for point in vector_results:

        payload = point.payload or {}

        chunk_id = str(
            payload.get("chunk_id", "")
        )

        if chunk_id not in chunk_lookup:
            continue

        if point.vector is None:
            raise ValueError(
                f"Missing vector for Qdrant point: {chunk_id}"
            )

        vector_candidates[chunk_id] = {
            "score": float(point.score),
            "vector": point.vector
        }

    # -------------------------------------------------
    # D. MERGE CANDIDATES BY CHUNK ID
    # -------------------------------------------------

    candidate_ids = list(
        dict.fromkeys(
            list(vector_candidates.keys())
            + list(bm25_candidates.keys())
        )
    )

    if not candidate_ids:
        print("No relevant chunks found.")
        return

    # -------------------------------------------------
    # E. RECIPROCAL RANK FUSION (RRF)
    # -------------------------------------------------

    # Record BM25 ranking positions
    bm25_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(
            bm25_candidates.keys(),
            start=1
        )
    }

    # Record Qdrant ranking positions
    vector_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(
            vector_candidates.keys(),
            start=1
        )
    }

    rrf_scores = []

    for chunk_id in candidate_ids:

        vector_rank = vector_ranks.get(chunk_id)

        bm25_rank = bm25_ranks.get(chunk_id)

        vector_rrf = (
            1 / (RRF_K + vector_rank)
            if vector_rank is not None
            else 0.0
        )

        bm25_rrf = (
            1 / (RRF_K + bm25_rank)
            if bm25_rank is not None
            else 0.0
        )

        combined_score = (
            alpha * vector_rrf
            + (1 - alpha) * bm25_rrf
        )

        rrf_scores.append(combined_score)

    rrf_scores = np.asarray(
        rrf_scores,
        dtype=np.float32
    )

    # Normalize RRF scores for MMR
    hybrid_scores = (
        rrf_scores / max(float(rrf_scores.max()), 1e-12)
    )

    # -------------------------------------------------
    # F. GET VECTORS FOR MMR
    # -------------------------------------------------

    candidate_vectors = []

    for chunk_id in candidate_ids:

        if chunk_id in vector_candidates:

            vector = vector_candidates[chunk_id]["vector"]

        else:

            # BM25-only candidates need vectors for MMR.
            # Retrieve their stored vectors using UUID5.

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    chunk_id
                )
            )

            point = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[point_id],
                with_vectors=True,
                with_payload=False
            )

            if not point or point[0].vector is None:
                raise ValueError(
                    f"Vector not found for chunk: {chunk_id}"
                )

            vector = point[0].vector

        candidate_vectors.append(vector)

    # -------------------------------------------------
    # G. APPLY MMR
    # -------------------------------------------------

    top_chunk_ids = mmr_search(
        candidate_ids=candidate_ids,
        relevance_scores=hybrid_scores,
        candidate_vectors=candidate_vectors,
        top_k=top_k,
        lambda_param=MMR_LAMBDA
    )

    # -------------------------------------------------
    # H. DISPLAY RESULTS
    # -------------------------------------------------

    print("\n========== QDRANT HYBRID + RRF + MMR RESULTS ==========")

    for rank, chunk_id in enumerate(
        top_chunk_ids,
        start=1
    ):

        chunk = chunk_lookup[chunk_id]

        index = candidate_ids.index(chunk_id)

        vector_score = (
            vector_candidates[chunk_id]["score"]
            if chunk_id in vector_candidates
            else 0.0
        )

        bm25_score = bm25_candidates.get(
            chunk_id,
            0.0
        )

        print(f"\nRank: {rank}")

        print(
            f"Normalized RRF Score: {hybrid_scores[index]:.4f}"
        )

        print(
            f"Vector Score: {vector_score:.4f}"
        )

        print(
            f"BM25 Score: {bm25_score:.4f}"
        )

        print(
            f"Source: {chunk.get('Source_Document')}"
        )

        print(
            f"Chunk ID: {chunk_id}"
        )

        # Limit console output
        text = chunk.get("Chunk", "")

        print(f"Text:\n{text[:1000]}")

        if len(text) > 1000:
            print("... [Text truncated in console]")

        print("-" * 70)

# =====================================================
# 9. RUN SEARCH
# =====================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       TELECOM QDRANT HYBRID SEARCH")
    print("========================================")

    while True:

        question = input(
            "\nEnter your question (or type exit): "
        )

        if question.lower().strip() == "exit":

            print("Exiting search system...")
            break

        try:
            hybrid_search(question)

        except Exception as error:
            print("\nSearch error:", error)
            print("Please check the error before continuing.")