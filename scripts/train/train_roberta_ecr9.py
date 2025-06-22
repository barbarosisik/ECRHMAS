import os
from transformers import RobertaTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput

ECR_LABELS = ['like', 'negative', 'curious', 'grateful', 'neutral', 'happy', 'surprise', 'nostalgia', 'agreement']

#emotion sample counts
label_counts = [9594, 9594, 4843, 2487, 4934, 7496, 3956, 4423, 3438]
total = sum(label_counts)
class_weights = [total / (len(label_counts) * c) for c in label_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

def get_dataset(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            samples.append({"text": d["text"], "label": int(d["label"])})
    return Dataset.from_list(samples)

def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

import torch.nn as nn
from transformers import RobertaForSequenceClassification

class WeightedRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

train_path = "ed_train_ecr9_train.jsonl"
valid_path = "ed_train_ecr9_valid.jsonl"
model_dir = "roberta_ecr9_classifier"

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_ds = get_dataset(train_path)
valid_ds = get_dataset(valid_path)
train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
valid_ds = valid_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = WeightedRobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(ECR_LABELS),
)
model.class_weights = class_weights_tensor

training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=2000,
    save_total_limit=1,
    logging_steps=100,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    report_to=[],
    dataloader_num_workers=4,
    disable_tqdm=False,
    no_cuda=(not torch.cuda.is_available()),
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    from sklearn.metrics import classification_report
    report = classification_report(labels, preds, target_names=ECR_LABELS, output_dict=True)
    overall_acc = report["accuracy"]
    #per-class results
    for lbl in ECR_LABELS:
        print(f"{lbl:10} precision: {report[lbl]['precision']:.3f}, recall: {report[lbl]['recall']:.3f}, f1: {report[lbl]['f1-score']:.3f}, support: {report[lbl]['support']}")
    print(f"Overall accuracy: {overall_acc:.3f}")
    return {"accuracy": overall_acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Training on device:", "cuda" if torch.cuda.is_available() else "cpu")

trainer.train()

with open(f"{model_dir}/label_names.txt", "w", encoding="utf-8") as f:
    for l in ECR_LABELS:
        f.write(f"{l}\n")
