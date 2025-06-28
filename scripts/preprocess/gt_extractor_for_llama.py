import json

TRAIN_FILE = "/data/s3905993/ECRHMAS/data/llama_train_converted.jsonl"
TEST_FILE = "/data/s3905993/ECRHMAS/data/llama_test_converted.jsonl"
OUTPUT_FILE = "/data/s3905993/ECRHMAS/data/llama_test_with_gt.jsonl"

# Helper: get a normalized string for context matching
def context_key(ctx):
    if isinstance(ctx, list) and ctx:
        # Take the first utterance, strip and lowercase for normalization
        return ctx[0].strip().lower()
    return ""

# 1. Load the train data and build a mapping from context to response
train_map = {}
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        key = context_key(item.get("context", []))
        if key:
            train_map[key] = item.get("resp", "")
import json

TRAIN_FILE = "/data/s3905993/ECRHMAS/data/llama_train_converted.jsonl"
TEST_FILE = "/data/s3905993/ECRHMAS/data/llama_test_converted.jsonl"
OUTPUT_FILE = "/data/s3905993/ECRHMAS/data/llama_test_with_gt.jsonl"

#getting a normalized string for context matching
def context_key(ctx):
    if isinstance(ctx, list) and ctx:
        #taking the first utterance, strip and lowercase for normalization
        return ctx[0].strip().lower()
    return ""

#loading the train data and build a mapping from context to response
train_map = {}
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        key = context_key(item.get("context", []))
        if key:
            train_map[key] = item.get("resp", "")

count = 0
with open(TEST_FILE, "r", encoding="utf-8") as f, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in f:
        item = json.loads(line)
        key = context_key(item.get("context", []))
        gt = train_map.get(key, "")
        if gt:
            count += 1
        item["ground_truth_response"] = gt
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Filled ground truth for {count} of {sum(1 for _ in open(TEST_FILE))} test samples.")
print(f"Saved to {OUTPUT_FILE}")
