import json
import re

DATA_PATH = "/data/s3905993/ECRMAS/ECR-main/src_emo/data/redial/train_data_processed.jsonl"
OUT_PATH = "/data/s3905993/ECRHMAS/src/data/train_intent_labeled.jsonl"

#heuristic for intent assignments
feedback_phrases = ["thank you", "thanks", "appreciate", "that's helpful", "helpful", "grateful"]
chitchat_phrases = ["hello", "hi", "how are you", "what's up", "goodbye", "bye", "see you"]
recommendation_phrases = [
    "recommend", "suggest", "can you recommend", "could you recommend", "would you recommend",
    "what should i watch", "do you know any good movies", "any movie", "movie to watch", "any recommendations"
]

def assign_intent(sample):
    resp = sample.get("resp", "").lower()
    #seeking_recommendation if 'rec' present and non-empty OR contains recommendation-request phrase
    if (sample.get("rec") and len(sample["rec"]) > 0) or any(phrase in resp for phrase in recommendation_phrases):
        return "seeking_recommendation"
    if any(phrase in resp for phrase in feedback_phrases):
        return "feedback"
    if any(phrase in resp for phrase in chitchat_phrases):
        return "chit_chat"
    return "other"

with open(DATA_PATH, "r") as fin, open(OUT_PATH, "w") as fout:
    for line in fin:
        data = json.loads(line)
        data["intent"] = assign_intent(data)
        fout.write(json.dumps(data) + "\n")

print(f"Intent-labeled file saved to: {OUT_PATH}")
