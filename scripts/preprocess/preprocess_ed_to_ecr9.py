import csv
import json
import random

input_csv = "/data/s3905993/ECRHMAS/data/empathetic_dialogues/train.csv"
output_jsonl = "ed_train_ecr9.jsonl"

ECR_LABELS = ['like', 'negative', 'curious', 'grateful', 'neutral', 'happy', 'surprise', 'nostalgia', 'agreement']
ED_TO_ECR_MAP = {
    "proud": "like",
    "impressed": "like",
    "confident": "like",
    "supportive": "like",
    "optimistic": "like",
    "inspired": "like",
    "caring": "like", 
    "appreciative": "grateful",
    "excited": "happy",
    "joyful": "happy",
    "happy": "happy",
    "glad": "happy",
    "cheerful": "happy",
    "elated": "happy",
    "delighted": "happy",
    "content": "happy",
    "grateful": "grateful",
    "thankful": "grateful",
    "appreciative": "grateful",
    "surprised": "surprise",
    "amazed": "surprise",
    "astonished": "surprise",
    "shocked": "surprise",
    "curious": "curious",
    "anticipating": "curious",
    "wondering": "curious",
    "inquisitive": "curious",
    "hopeful": "curious",
    "confused": "curious",
    "faithful": "agreement",
    "trusting": "agreement",
    "agreeing": "agreement",
    "cooperative": "agreement",
    "supportive": "agreement",
    "nostalgic": "nostalgia",
    "sentimental": "nostalgia",
    "melancholy": "nostalgia",
    "homesick": "nostalgia",
    "wistful": "nostalgia",
    "neutral": "neutral",
    "bored": "neutral",
    "uninterested": "neutral",
    "indifferent": "neutral",
    "apathetic": "neutral",
    "afraid": "negative",
    "sad": "negative",
    "angry": "negative",
    "guilty": "negative",
    "disgusted": "negative",
    "anxious": "negative",
    "lonely": "negative",
    "embarrassed": "negative",
    "disappointed": "negative",
    "furious": "negative",
    "remorseful": "negative",
    "terrified": "negative",
    "jealous": "negative",
    "devastated": "negative",
    "ashamed": "negative",
    "miserable": "negative",
    "frustrated": "negative",
    "worried": "negative",
    "apprehensive": "negative",
    "hurt": "negative",
    "unhappy": "negative",
    "rejected": "negative",
    "resentful": "negative",
    "hopeless": "negative",
    "empty": "negative",
}

#collecting all samples by labels
samples_by_label = {k: [] for k in ECR_LABELS}
with open(input_csv, encoding="utf-8") as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        text = row["utterance"]
        ed_label = row["context"].strip().lower()
        ecr_label = ED_TO_ECR_MAP.get(ed_label, "neutral")
        if text and ecr_label in ECR_LABELS:
            samples_by_label[ecr_label].append({"text": text, "label": ECR_LABELS.index(ecr_label)})

#downsampling "negative" samples to match "like" count, since there was too much negatives
target_count = len(samples_by_label["like"])
random.seed(42)

neg_samples = samples_by_label["negative"]
if len(neg_samples) > target_count:
    neg_samples = random.sample(neg_samples, target_count)
samples_by_label["negative"] = neg_samples

#combining all balanced samples
balanced_samples = []
label_count = {}
for label in ECR_LABELS:
    balanced_samples.extend(samples_by_label[label])
    label_count[label] = len(samples_by_label[label])

with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for entry in balanced_samples:
        f_out.write(json.dumps(entry) + "\n")

with open("label_names.txt", "w", encoding="utf-8") as f:
    for l in ECR_LABELS:
        f.write(f"{l}\n")

print(label_count)