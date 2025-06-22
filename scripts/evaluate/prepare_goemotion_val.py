from datasets import load_dataset
import json

<<<<<<< HEAD
#loading GoEmotions validation split
dataset = load_dataset("go_emotions", split="validation")

=======
#GoEmotions validation split
dataset = load_dataset("go_emotions", split="validation")

#ClassLabel object
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
class_labels = dataset.features['labels'].feature

#preparing samples
samples = []
for example in dataset:
    if example['labels']:
        label_id = example['labels'][0]
        label_name = class_labels.int2str(label_id)
        samples.append({'text': example['text'], 'label': label_name})

#saving
with open("emotion_val_set.json", "w") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"Saved {len(samples)} samples to emotion_val_set.json")
