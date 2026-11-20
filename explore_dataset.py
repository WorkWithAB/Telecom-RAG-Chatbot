
from datasets import load_dataset
import pandas as pd

print("Loading Telecom Dataset...")

dataset = load_dataset("MantisNLP/Telecom-Questions")

df = dataset["train"].to_pandas()

print("\nDataset Loaded Successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nQuestion Types:")
print(df["Question_Type"].value_counts())

print("\nDataset Information:")
print(df.info())

# =====================================
# DATASET ANALYSIS
# =====================================

print("\n========== DATASET ANALYSIS ==========")

# 1. Dataset dimensions
print("Total Questions:", len(df))

# 2. Duplicate questions
print("Duplicate Questions:", df["Question"].duplicated().sum())

# 3. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Question type distribution
print("\nQuestion Type Distribution:")
print(df["Question_Type"].value_counts())

# 5. Working group distribution
print("\nWorking Group Distribution:")
print(df["Working_Group"].value_counts().head(10))

# 6. Average question length
df["question_length"] = df["Question"].astype(str).apply(len)

print("\nAverage Question Length:")
print(df["question_length"].mean())

# 7. Average answer length
df["answer_length"] = df["Answer"].astype(str).apply(len)

print("\nAverage Answer Length:")
print(df["answer_length"].mean())

# 8. Display one sample
print("\n========== SAMPLE RECORD ==========")
print("Question:", df.iloc[0]["Question"])
print("\nAnswer:", df.iloc[0]["Answer"])
print("\nChunk:", df.iloc[0]["Chunk"])

# =====================================
# DUPLICATE INVESTIGATION
# =====================================

print("\n========== DUPLICATE INVESTIGATION ==========")

# Find duplicate questions
duplicates = df[
    df["Question"].duplicated(keep=False)
].sort_values("Question")

print("Total duplicate-question records:", len(duplicates))

# Show first 10 duplicate groups
print("\nSample duplicate questions:\n")

for question, group in duplicates.groupby("Question"):
    print("\nQUESTION:", question)
    print("Number of occurrences:", len(group))

    for index, row in group.head(3).iterrows():
        print("Answer:", row["Answer"][:200])
        print("Source:", row["Source_Document"])
        print("Chunk ID:", row["Chunk_ID"])

    if question == duplicates["Question"].iloc[-1]:
        break

# Check completely identical records
print("\nCompletely identical rows:",
      df.duplicated().sum())

# Analyze chunk lengths
df["chunk_length"] = df["Chunk"].astype(str).apply(len)

print("\n========== CHUNK ANALYSIS ==========")
print(df["chunk_length"].describe())

# =====================================
# FINAL DATA QUALITY CHECK
# =====================================

print("\n========== DUPLICATE QUESTION CHECK ==========")

question_groups = (
    df.groupby("Question")
    .agg(
        records=("Question", "size"),
        unique_answers=("Answer", "nunique"),
        unique_sources=("Source_Document", "nunique")
    )
    .reset_index()
)

repeated = question_groups[question_groups["records"] > 1]

print("Repeated question groups:", len(repeated))

print(
    "Questions with different answers:",
    len(repeated[repeated["unique_answers"] > 1])
)

print(
    "Questions appearing in different documents:",
    len(repeated[repeated["unique_sources"] > 1])
)

print("\nExamples of questions with different answers:")

conflicts = repeated[repeated["unique_answers"] > 1]

for _, item in conflicts.head(10).iterrows():
    print("\nQuestion:", item["Question"])
    print("Records:", item["records"])
    print("Different answers:", item["unique_answers"])

# =====================================
# CHUNK UNIQUENESS CHECK
# =====================================

print("\n========== CHUNK UNIQUENESS ==========")

chunk_groups = (
    df.groupby(["Source_Document", "Chunk_ID"])
    .agg(
        records=("Chunk_ID", "size"),
        unique_chunks=("Chunk", "nunique")
    )
    .reset_index()
)

print("Unique source-document/chunk-ID combinations:",
      len(chunk_groups))

print("Combinations with multiple records:",
      len(chunk_groups[chunk_groups["records"] > 1]))

print("Combinations with different chunk text:",
      len(chunk_groups[chunk_groups["unique_chunks"] > 1]))

# =====================================
# UNIQUE CHUNK ANALYSIS
# =====================================

print("\n========== UNIQUE CHUNK ANALYSIS ==========")

# Keep source document and actual chunk text
unique_chunks = df[
    ["Source_Document", "Chunk"]
].drop_duplicates()

print("Total original records:", len(df))

print("Unique chunks:", len(unique_chunks))

print(
    "Duplicate chunk records removed:",
    len(df) - len(unique_chunks)
)

# Calculate chunk lengths
unique_chunks = unique_chunks.copy()

unique_chunks["chunk_length"] = (
    unique_chunks["Chunk"].astype(str).str.len()
)

print("\nUnique chunk length statistics:")

print(
    unique_chunks["chunk_length"].describe()
)

# Check empty chunks
print(
    "\nEmpty chunks:",
    (unique_chunks["Chunk"].astype(str).str.strip() == "").sum()
)
