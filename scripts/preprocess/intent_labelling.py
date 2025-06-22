import json
from tqdm import tqdm

input_path = "/data/s3905993/ECRHMAS/data/redial/train_data_processed.jsonl"
output_path = "/data/s3905993/ECRHMAS/data/redial/intent_train.jsonl"
label_names = ["seeking_recommendation", "feedback", "chit_chat", "other"]

def label_intent(entry):
    resp = entry.get("resp", "").lower()
    role = entry.get("role", "").lower()
    recs = entry.get("rec", [])
    #chitchat
    if any(kw in resp for kw in ["hello", "hi", "how are you", "bye", "goodbye"]):
        return "chit_chat"
    #seeking recomm
    if role == "seeker" and ("?" in resp or "recommend" in resp):
        return "seeking_recommendation"
    #feedback
    if role == "seeker" and any(w in resp for w in ["thanks", "thank", "like", "love", "enjoyed", "hated", "dislike"]):
        return "feedback"
    #if there is the rec 
    if role == "recommender" and recs and len(recs) > 0:
        return "feedback"
    #fallback
    return "other"

with open(input_path, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
    for line in tqdm(f_in):
        entry = json.loads(line)
        label = label_intent(entry)
        if label is not None:
            label_id = label_names.index(label)
            f_out.write(json.dumps({"text": entry["resp"], "label": label_id}) + "\n")

with open("/data/s3905993/ECRHMAS/data/redial/intent_label_names.json", "w", encoding="utf-8") as f:
    json.dump(label_names, f)
