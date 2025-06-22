import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json
import os

MODEL_PATH = "/data/s3905993/ECRHMAS/src/models/intent_classifier"

<<<<<<< HEAD
=======
#loading label mapping
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
with open(os.path.join(MODEL_PATH, "intent_labels.json")) as f:
    INTENT_LABELS = json.load(f)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

examples = [
    "Could you recommend me a feel-good movie?",
    "Thanks for your help!",
    "Hey, how are you doing?",
    "I saw that movie already.",
]

for text in examples:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs))
    pred_label = INTENT_LABELS[pred_idx]
    print(f"Text: {text}\nPredicted Intent: {pred_label}\nScores: {probs.tolist()}\n")
