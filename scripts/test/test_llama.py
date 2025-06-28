from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data/s3905993/ECRHMAS/src/models/llama2_chat/"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Loading model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(model_path)
print("Loaded successfully.")
