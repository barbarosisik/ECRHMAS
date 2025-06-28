import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch

#params
MODEL_NAME_OR_PATH = "/data/s3905993/ECRHMAS/src/models/llama2_chat"
TRAIN_FILE = "/data/s3905993/ECRHMAS/data/llama_train_converted.jsonl"
OUTPUT_DIR = "/data/s3905993/ECRHMAS/models/llama2_finetuned_movie"
MAX_SEQ_LENGTH = 512 
BATCH_SIZE = 3
NUM_EPOCHS = 3 
LEARNING_RATE = 2e-5

#load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
tokenizer.pad_token = tokenizer.eos_token

#dataset preperation
def build_prompt(sample):
    """Format the prompt as you do at inference time."""
    context = sample["context"]
    knowledge = sample.get("knowledge_used", "")
    #concat context lines
    dialogue = "\n".join(context)
    prompt = (
        "<s>[INST] <<SYS>>\n"
        "You are an empathetic movie assistant. Always mention the movie title and adapt your tone to the user's emotion. Be concise and empathetic.\n"
        "<</SYS>>\n"
        f"{dialogue}\n"
    )
    if knowledge:
        prompt += f"[KNOWLEDGE]\n{knowledge}\n"
    prompt += "\nAssistant:"
    return prompt

def preprocess(example):
    prompt = build_prompt(example)
    response = example["resp"].strip()
    #full example is prompt + response as Llama is trained in next-token prediction.
    full_text = prompt + " " + response + tokenizer.eos_token
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  #lm task
    return tokenized

#load and preprocess the dataset
print("Loading dataset...")
raw_dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE},
    split="train"
)
print(f"Loaded {len(raw_dataset)} training samples.")

print("Tokenizing...")
dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)

#training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    bf16=False,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to=[], 
    evaluation_strategy="no",
    disable_tqdm=False,
    dataloader_num_workers=2,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

#trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#training
print("Starting training...")
trainer.train()
print("Training complete. Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuned Llama model saved to {OUTPUT_DIR}")
