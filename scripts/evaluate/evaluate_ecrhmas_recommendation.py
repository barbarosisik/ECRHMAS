import json
import torch

K_LIST = [1, 10, 50]  #change for recall@k, mrr@k, ndcg@k
INPUT_FILE = "/data/s3905993/ECRHMAS/results/ecrhmas_ecr9_full.jsonl"

metrics = {f"recall@{k}": 0 for k in K_LIST}
metrics.update({f"mrr@{k}": 0 for k in K_LIST})
metrics.update({f"ndcg@{k}": 0 for k in K_LIST})
metrics["count"] = 0

def compute_recall(pred, label, k):
    return int(label in pred[:k])

def compute_mrr(pred, label, k):
    if label in pred[:k]:
        return 1 / (pred.index(label) + 1)
    return 0

def compute_ndcg(pred, label, k):
    if label in pred[:k]:
        return 1 / torch.log2(torch.tensor(pred.index(label) + 2).float()).item()
    return 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        sample = json.loads(line)
        #getting predicted movie ids
        recommended_items = sample.get("recommended_items", [])
        pred_ids = []
        for rec in recommended_items:
            if isinstance(rec, dict):
                pred_ids.append(rec.get("movie_id"))
            elif isinstance(rec, int):
                pred_ids.append(rec)

        #ground truth
        label = sample.get("ground_truth_rec", None)
        if label is None or not pred_ids:
            continue
        try:
            label = int(label)
            pred_ids = [int(x) for x in pred_ids if x is not None]
        except Exception:
            continue

        for k in K_LIST:
            metrics[f"recall@{k}"] += compute_recall(pred_ids, label, k)
            metrics[f"mrr@{k}"] += compute_mrr(pred_ids, label, k)
            metrics[f"ndcg@{k}"] += compute_ndcg(pred_ids, label, k)
        metrics["count"] += 1

#normalizing
for k in K_LIST:
    for metric_type in ["recall", "mrr", "ndcg"]:
        key = f"{metric_type}@{k}"
        metrics[key] = metrics[key] / metrics["count"] if metrics["count"] > 0 else 0

#report
print("==== HMAS Recommendation Evaluation ====")
for k in K_LIST:
    print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
    print(f"MRR@{k}:    {metrics[f'mrr@{k}']:.4f}")
    print(f"NDCG@{k}:   {metrics[f'ndcg@{k}']:.4f}")
print(f"Total evaluated: {metrics['count']}")
