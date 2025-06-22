import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from tqdm import tqdm
from sklearn.metrics import classification_report

<<<<<<< HEAD
#settings
TRAIN_PATH = "/data/s3905993/ECRHMAS/data/redial/intent_train_train.jsonl"
VALID_PATH = "/data/s3905993/ECRHMAS/data/redial/intent_train_valid.jsonl"
LABELS_PATH = "/data/s3905993/ECRHMAS/data/redial/intent_label_names.json"
MODEL_SAVE_PATH = "/data/s3905993/ECRHMAS/src/models/roberta_intent_classifier"
=======
#SETTINGS
DATA_PATH = "/data/s3905993/ECRHMAS/src/data/train_intent_labeled.jsonl"
MODEL_SAVE_PATH = "/data/s3905993/ECRHMAS/src/models/intent_classifier"
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 128

<<<<<<< HEAD
#loading labels
with open(LABELS_PATH, "r") as f:
    INTENT_LABELS = json.load(f)
=======
#defining intent label set
INTENT_LABELS = ["seeking_recommendation", "feedback", "chit_chat", "other"]
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
label2idx = {label: i for i, label in enumerate(INTENT_LABELS)}
idx2label = {i: label for label, i in label2idx.items()}

#dataset
class IntentDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        self.samples = []
        with open(json_path, "r") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "").strip()
                label = int(data.get("label", -1))
                if text and label != -1:
                    self.samples.append((text, label))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoded = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_true = batch["labels"].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            y_pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(y_pred)
            labels.extend(y_true)
    model.train()
    print(classification_report(labels, preds, target_names=INTENT_LABELS, digits=3))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(INTENT_LABELS)).to(device)

    train_dataset = IntentDataset(TRAIN_PATH, tokenizer)
    valid_dataset = IntentDataset(VALID_PATH, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * EPOCHS
    )

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dataloader):.4f}")
        #eval on each epoch for debug
        print(f"Validation results after epoch {epoch+1}:")
        evaluate(model, valid_dataloader, device)

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
<<<<<<< HEAD
=======

>>>>>>> fa533849599714b8f490bc557652c007aa45e497
    with open(os.path.join(MODEL_SAVE_PATH, "intent_labels.json"), "w") as f:
        json.dump(INTENT_LABELS, f)
    print(f"Model and label mapping saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
