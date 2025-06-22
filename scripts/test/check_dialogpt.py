from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/data/s3905993/ECRHMAS/src/models/emotion_dialoGPT"
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

prompt = "Hello, I am feeling happy because"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=20,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated response:", response)
