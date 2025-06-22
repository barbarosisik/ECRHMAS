from mas_agent1 import EmotionIntentRecognizerMAS

intent_model_path = "/data/s3905993/ECRHMAS/src/models/intent_classifier"
INTENT_LABELS = ["seeking_recommendation", "feedback", "chit_chat", "other"]

agent = EmotionIntentRecognizerMAS(
    intent_model_path=intent_model_path,
    intent_label_list=INTENT_LABELS,
    history_turns=3,
    debug=True
)

dialogue = [
    "System: What kind of movies do you like?",
    "User: I like movies that takes place in space",
    "System: You mean movies like Star Wars or Interstellar?",
    "User: Exactly like Interstaller! Could you tell me a movie that is like Interstellar?"
]
gold_emotion_label = ""

result = agent.process(dialogue, gold_emotion_label)
print(result)
