# import json
# import re
# from nltk import ngrams
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# INPUT_FILE = "/data/s3905993/ECRHMAS/results/ecrhmas_ecr9_full.jsonl"
# LOG_FILE = "/data/s3905993/ECRHMAS/results/response_eval_hmas.log"

# metrics = {
#     'bleu@1': 0,
#     'bleu@2': 0,
#     'bleu@3': 0,
#     'bleu@4': 0,
#     'dist@1': set(),
#     'dist@2': set(),
#     'dist@3': set(),
#     'dist@4': set(),
#     'item_ratio': 0,
#     'sent_count': 0,
# }

# slot_pattern = re.compile(r"<movie>", re.IGNORECASE)

# def tokenize_for_bleu(text):
#     return text.strip().split()

# def compute_bleu(pred, label):
#     scores = []
#     pred_tokens = tokenize_for_bleu(pred)
#     label_tokens = [tokenize_for_bleu(label)]
#     smoothing = SmoothingFunction().method1
#     for i in range(4):
#         weights = [0] * 4
#         weights[i] = 1
#         score = sentence_bleu(label_tokens, pred_tokens, weights, smoothing_function=smoothing)
#         scores.append(score)
#     return scores

# def compute_distinct_ngrams(tokens, n):
#     return set(ngrams(tokens, n))

# def evaluate(preds, labels, contexts=None):
#     with open(LOG_FILE, "w", encoding="utf-8") as log_file:
#         for pred, label, ctxt in zip(preds, labels, contexts):
#             if not pred or not pred.strip():
#                 continue

#             metrics['sent_count'] += 1

#             #logging
#             log_file.write(json.dumps({
#                 "input": ctxt,
#                 "pred": pred,
#                 "label": label
#             }, ensure_ascii=False) + "\n")

#             #BLEU
#             b1, b2, b3, b4 = compute_bleu(pred, label)
#             metrics['bleu@1'] += b1
#             metrics['bleu@2'] += b2
#             metrics['bleu@3'] += b3
#             metrics['bleu@4'] += b4

#             #distinct-N
#             tokens = tokenize_for_bleu(pred)
#             for n in range(1, 5):
#                 metrics[f'dist@{n}'].update(compute_distinct_ngrams(tokens, n))

#             #item ratio
#             metrics['item_ratio'] += len(slot_pattern.findall(pred))

# #loading data
# with open(INPUT_FILE, "r", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]

# pred_texts = [item.get("response", "") for item in data]
# ref_texts = [item.get("ground_truth_response", "") for item in data]
# contexts = [item.get("context", []) for item in data]

# evaluate(pred_texts, ref_texts, contexts)

# #report
# print("==== HMAS Response Evaluation ====")
# for k, v in metrics.items():
#     if k.startswith("dist@"):
#         print(f"{k}: {len(v) / metrics['sent_count']:.4f}" if metrics['sent_count'] else f"{k}: 0.0000")
#     elif k.startswith("bleu@"):
#         print(f"{k}: {v / metrics['sent_count']:.4f}" if metrics['sent_count'] else f"{k}: 0.0000")
#     elif k == "item_ratio":
#         print(f"{k}: {v / metrics['sent_count']:.4f}" if metrics['sent_count'] else f"{k}: 0.0000")
#     elif k == "sent_count":
#         print(f"{k}: {v}")

# import json
# import re
# from nltk import ngrams
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# INPUT_FILE = "/data/s3905993/ECRHMAS/results/ecrhmas_ecr9_full.jsonl"
# LOG_FILE = "/data/s3905993/ECRHMAS/results/response_eval_hmas.log"

# metrics = {
#     'bleu@1': 0,
#     'bleu@2': 0,
#     'bleu@3': 0,
#     'bleu@4': 0,
#     'dist@1': set(),
#     'dist@2': set(),
#     'dist@3': set(),
#     'dist@4': set(),
#     'item_ratio': 0,
#     'sent_count': 0,
# }

# slot_pattern = re.compile(r"<movie>", re.IGNORECASE)

# def tokenize_for_bleu(text):
#     return text.strip().split()

# def compute_bleu(pred, label):
#     scores = []
#     pred_tokens = tokenize_for_bleu(pred)
#     label_tokens = [tokenize_for_bleu(label)]
#     smoothing = SmoothingFunction().method1
#     for i in range(4):
#         weights = [0] * 4
#         weights[i] = 1
#         score = sentence_bleu(label_tokens, pred_tokens, weights, smoothing_function=smoothing)
#         scores.append(score)
#     return scores

# def compute_distinct_ngrams(tokens, n):
#     return set(ngrams(tokens, n))

# def evaluate(preds, labels, contexts=None):
#     with open(LOG_FILE, "w", encoding="utf-8") as log_file:
#         for pred, label, ctxt in zip(preds, labels, contexts):
#             if not pred or not pred.strip():
#                 continue

#             metrics['sent_count'] += 1

#             #logging
#             log_file.write(json.dumps({
#                 "input": ctxt,
#                 "pred": pred,
#                 "label": label
#             }, ensure_ascii=False) + "\n")

#             #BLEU
#             b1, b2, b3, b4 = compute_bleu(pred, label)
#             metrics['bleu@1'] += b1
#             metrics['bleu@2'] += b2
#             metrics['bleu@3'] += b3
#             metrics['bleu@4'] += b4

#             #distinct-N
#             tokens = tokenize_for_bleu(pred)
#             for n in range(1, 5):
#                 metrics[f'dist@{n}'].update(compute_distinct_ngrams(tokens, n))

#             #item ratio
#             metrics['item_ratio'] += len(slot_pattern.findall(pred))

# #loading data
# with open(INPUT_FILE, "r", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]

# pred_texts = [item.get("response", "") for item in data]
# ref_texts = [item.get("ground_truth_response", "") for item in data]
# contexts = [item.get("context", []) for item in data]

# evaluate(pred_texts, ref_texts, contexts)

# #report
# print("==== HMAS Response Evaluation ====")
# for k, v in metrics.items():
#     if k.startswith("dist@"):
#         print(f"{k}: {len(v) / metrics['sent_count']:.4f}" if metrics['sent_count'] else f"{k}: 0.0000")
#     elif k.startswith("bleu@"):
#         print(f"{k}: {v / metrics['sent_count']:.4f}" if metrics['sent_count'] else f"{k}: 0.0000")
#     elif k == "item_ratio":
#         print(f"{k}: {v / metrics['sent_count']:.4f}" if metrics['sent_count'] else f"{k}: 0.0000")
#     elif k == "sent_count":
#         print(f"{k}: {v}")

import json
import re
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load movie titles for normalization
with open("/data/s3905993/ECRHMAS/data/redial_gen/movie_name.json", "r") as f:
    MOVIE_TITLES = json.load(f)

def replace_movie_names(text, movie_titles):
    for title in sorted(movie_titles, key=lambda x: -len(x)):
        pattern = r'\b{}\b'.format(re.escape(title))
        text = re.sub(pattern, "<movie>", text)
    return text

INPUT_FILE = "/data/s3905993/ECRHMAS/results/ecrhmas_infer_llama.jsonl"
LOG_FILE = "/data/s3905993/ECRHMAS/results/response_eval_llama_infer_500.log"

metrics = {
    'bleu@1': 0,
    'bleu@2': 0,
    'bleu@3': 0,
    'bleu@4': 0,
    'item_ratio': 0,
    'sent_count': 0,
    #dist-n
    'distinct_counts': {n: set() for n in range(1, 5)},
    'distinct_totals': {n: 0 for n in range(1, 5)},
}

slot_pattern = re.compile(r"<movie>", re.IGNORECASE)

def tokenize_for_bleu(text):
    return text.strip().split()

def compute_bleu(pred, label):
    scores = []
    pred_tokens = tokenize_for_bleu(pred)
    label_tokens = [tokenize_for_bleu(label)]
    smoothing = SmoothingFunction().method1
    for i in range(4):
        weights = [0] * 4
        weights[i] = 1
        score = sentence_bleu(label_tokens, pred_tokens, weights, smoothing_function=smoothing)
        scores.append(score)
    return scores

def evaluate(preds, labels, contexts=None):
    with open(LOG_FILE, "w", encoding="utf-8") as log_file:
        for pred, label, ctxt in zip(preds, labels, contexts):
            if not pred or not pred.strip() or not label or not label.strip():
                continue

            metrics['sent_count'] += 1

            #logging
            log_file.write(json.dumps({
                "input": ctxt,
                "pred": pred,
                "label": label
            }, ensure_ascii=False) + "\n")

            #BLEU
            b1, b2, b3, b4 = compute_bleu(pred, label)
            metrics['bleu@1'] += b1
            metrics['bleu@2'] += b2
            metrics['bleu@3'] += b3
            metrics['bleu@4'] += b4

            #dist-n
            tokens = tokenize_for_bleu(pred)
            for n in range(1, 5):
                ngram_list = list(ngrams(tokens, n)) if len(tokens) >= n else []
                metrics['distinct_totals'][n] += max(len(tokens) - n + 1, 0)
                metrics['distinct_counts'][n].update(ngram_list)

            #item ratio
            metrics['item_ratio'] += len(slot_pattern.findall(pred))

#load data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

pred_texts = [replace_movie_names(item.get("response", ""), MOVIE_TITLES) for item in data]
ref_texts  = [replace_movie_names(item.get("ground_truth_response", ""), MOVIE_TITLES) for item in data]
contexts = [item.get("context", []) for item in data]

evaluate(pred_texts, ref_texts, contexts)

#print results
print("==== HMAS Response Evaluation ====")
for n in range(1, 5):
    total = metrics['distinct_totals'][n]
    uniq = len(metrics['distinct_counts'][n])
    print(f"Dist-{n}: {uniq / total:.4f}" if total else f"Dist-{n}: 0.0000")
for k in ['bleu@1', 'bleu@2', 'bleu@3', 'bleu@4']:
    print(f"{k}: {metrics[k] / metrics['sent_count']:.4f}" if metrics['sent_count'] else f"{k}: 0.0000")
print(f"item_ratio: {metrics['item_ratio'] / metrics['sent_count']:.4f}" if metrics['sent_count'] else "item_ratio: 0.0000")
print(f"sent_count: {metrics['sent_count']}")
