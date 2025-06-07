import json
from mas_agent1 import EmotionIntentRecognizerMAS

intent_model_path = "/data/s3905993/ECRHMAS/src/models/intent_classifier"
INTENT_LABELS = ["seeking_recommendation", "feedback", "chit_chat", "other"]

agent = EmotionIntentRecognizerMAS(
    intent_model_path=intent_model_path,
    intent_label_list=INTENT_LABELS,
    history_turns=3,
    debug=True
)

def get_gold_emotion(sample):
    """
    Extracts the gold emotion label from the last turn's emotion_entity list.
    Returns '' if not available.
    """
    # 'emotion_entity' is a list of lists; each sub-list is emotions for a turn
    if "emotion_entity" in sample and sample["emotion_entity"]:
        last_emotions = sample["emotion_entity"][-1]
        if isinstance(last_emotions, list) and last_emotions:
            return last_emotions[0]  # Pick the most probable (first) emotion
        elif isinstance(last_emotions, str):
            return last_emotions
    return ""

data_path = "/data/s3905993/ECRHMAS/data/redial/train_data_processed.jsonl"

with open(data_path) as fin:
    for idx, line in enumerate(fin):
        sample = json.loads(line)
        dialogue = sample["context"] + [sample["resp"]]
        gold_emotion = get_gold_emotion(sample)
        result = agent.process(dialogue, gold_emotion)
        print(f"Sample {idx}: {result}")
        if idx >= 299:  # Show only first 10 samples
            break
