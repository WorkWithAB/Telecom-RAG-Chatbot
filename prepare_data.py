
import pandas as pd
import hashlib
from pathlib import Path
from datasets import load_dataset

print("Loading telecom dataset...")

dataset = load_dataset("MantisNLP/Telecom-Questions")
df = dataset["train"].to_pandas()

print("Original records:", len(df))

# Keep required fields
unique_chunks = df[
    ["Source_Document", "Chunk", "Chunk_ID"]
].copy()

# Preserve original IDs
unique_chunks = unique_chunks.rename(
    columns={"Chunk_ID": "Original_Chunk_ID"}
)

# Remove empty chunks
unique_chunks = unique_chunks.dropna(
    subset=["Chunk", "Source_Document"]
)

unique_chunks["Chunk"] = (
    unique_chunks["Chunk"].astype(str).str.strip()
)

unique_chunks = unique_chunks[
    unique_chunks["Chunk"] != ""
]

# Remove duplicate text within the same document
unique_chunks = unique_chunks.drop_duplicates(
    subset=["Source_Document", "Chunk"]
).reset_index(drop=True)

# Generate reliable unique IDs
def generate_chunk_id(row):

    content = (
        str(row["Source_Document"])
        + "::"
        + str(row["Chunk"])
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

unique_chunks["Chunk_ID"] = unique_chunks.apply(
    generate_chunk_id,
    axis=1
)

# Add chunk length
unique_chunks["chunk_length"] = (
    unique_chunks["Chunk"].str.len()
)

# Create data folder
Path("data").mkdir(exist_ok=True)

# Save prepared dataset
output_path = "data/unique_chunks.jsonl"

unique_chunks.to_json(
    output_path,
    orient="records",
    lines=True,
    force_ascii=False
)

print("\n========== DATA PREPARATION COMPLETE ==========")

print("Unique chunks:", len(unique_chunks))

print(
    "Unique IDs:",
    unique_chunks["Chunk_ID"].nunique()
)

print(
    "Duplicate IDs:",
    unique_chunks["Chunk_ID"].duplicated().sum()
)

print("Saved file:", output_path)

print("\nFirst chunk ID:")
print(unique_chunks.iloc[0]["Chunk_ID"])
