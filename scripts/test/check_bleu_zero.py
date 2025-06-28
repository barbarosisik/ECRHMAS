import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

input_file = "/data/s3905993/ECRHMAS/results/ecrhmas_ecr9_full.jsonl"
with open(input_file) as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        ref = d.get("ground_truth_response", "")
        pred = d.get("response", "")
        if ref and pred:
            score = sentence_bleu([ref.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
            if score < 1e-3:
                print(f"\nREF: {ref}\nSYS: {pred}\n---")
                if i > 20: break
