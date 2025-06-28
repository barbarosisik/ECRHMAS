import json

TRAIN_FILE = "/data/s3905993/ECRHMAS/data/llama_train_converted.jsonl"
TEST_FILE = "/data/s3905993/ECRHMAS/data/llama_test_converted.jsonl"

train_contexts = set()
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        # normalize by lowercasing and stripping spaces
        key = item["context"][0].strip().lower()
        train_contexts.add(key)

matches = 0
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        key = item["context"][0].strip().lower()
        if key in train_contexts:
            print("MATCH FOUND:", key)
            matches += 1

print(f"Total test samples: {sum(1 for _ in open(TEST_FILE))}")
print(f"Matched contexts: {matches}")
