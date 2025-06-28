import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaResponseAgent:
    def __init__(self, model_dir, device='cuda', debug=False):
        self.debug = debug
        self.device = torch.device("cpu" if device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("Loading model (this may take a while)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        print("Loaded successfully.")

    def generate(self, prompt, max_new_tokens=120, temperature=0.7, top_p=0.85):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=4
            )
        decoded = self.tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        if self.debug:
            print("\n=== PROMPT ===\n", prompt)
            print("\n=== RESPONSE ===\n", decoded)
        return decoded.strip()
