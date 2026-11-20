
import json

file_path = "data/unique_chunks.jsonl"

print("Inspecting prepared dataset...\n")

with open(file_path, "r", encoding="utf-8") as file:

    for index, line in enumerate(file):

        if index >= 3:
            break

        record = json.loads(line)

        print(f"\n========== RECORD {index + 1} ==========")

        print("\nAvailable fields:")
        print(list(record.keys()))

        print("\nRecord information:")

        for key, value in record.items():

            print(f"\n{key}:")

            # Avoid printing extremely long text
            print(str(value)[:500])
