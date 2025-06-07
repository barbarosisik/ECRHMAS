from datasets import load_dataset
import json

# Load GoEmotions validation split
dataset = load_dataset("go_emotions", split="validation")

# Get ClassLabel object
class_labels = dataset.features['labels'].feature

# Prepare samples (first label only for single-label eval)
samples = []
for example in dataset:
    if example['labels']:
        label_id = example['labels'][0]   # use first label (single-label eval)
        label_name = class_labels.int2str(label_id)
        samples.append({'text': example['text'], 'label': label_name})

# Save as JSON
with open("emotion_val_set.json", "w") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"Saved {len(samples)} samples to emotion_val_set.json")
