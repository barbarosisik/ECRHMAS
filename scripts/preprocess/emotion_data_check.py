import json
from collections import Counter

ECR_LABELS = ['like', 'negative', 'curious', 'grateful', 'neutral', 'happy', 'surprise', 'nostalgia', 'agreement']

label_count = Counter()
with open('ed_train_ecr9.jsonl', encoding='utf-8') as f:   # <--- CHANGE THIS LINE
    for line in f:
        data = json.loads(line)
        label = data['label']
        label_count[label] += 1

print("Label distribution:")
for idx, label in enumerate(ECR_LABELS):
    print(f"{label:10}: {label_count[idx]}")
