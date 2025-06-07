import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_SET_PATH = "emotion_val_set.json"

MODELS = {
    "SamLowe/roberta-base-go_emotions": "SamLowe/roberta-base-go_emotions",
    "monologg/bert-base-cased-goemotions-original": "monologg/bert-base-cased-goemotions-original"
}

def load_data(path):
    with open(path) as f:
        return json.load(f)

def evaluate_model(model_name, samples):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    texts = [sample["text"] for sample in samples]
    true_labels = [sample["label"].lower() for sample in samples]
    preds = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_idx = logits.argmax(-1).item()
        pred_label = model.config.id2label[pred_idx].lower()
        preds.append(pred_label)

    report = classification_report(true_labels, preds, zero_division=0)
    print(f"\n--- Model: {model_name} ---\n{report}")

def main():
    samples = load_data(TEST_SET_PATH)
    for model_key, model_path in MODELS.items():
        print(f"Evaluating model: {model_key}")
        evaluate_model(model_path, samples)

if __name__ == "__main__":
    main()
