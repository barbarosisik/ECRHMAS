import json
import random

input_path = "ed_train_ecr9.jsonl"
train_path = "ed_train_ecr9_train.jsonl"
valid_path = "ed_train_ecr9_valid.jsonl"
valid_ratio = 0.10

with open(input_path, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.seed(42)
random.shuffle(data)
n_valid = int(len(data) * valid_ratio)

valid_data = data[:n_valid]
train_data = data[n_valid:]

with open(train_path, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(valid_path, "w", encoding="utf-8") as f:
    for item in valid_data:
        f.write(json.dumps(item) + "\n")

print(f"Train: {len(train_data)} samples")
print(f"Valid: {len(valid_data)} samples")
