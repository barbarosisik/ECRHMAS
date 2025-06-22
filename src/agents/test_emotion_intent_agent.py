from emotion_intent_recognizer_agent import EmotionIntentRecognizer

#example labels needs to align with the original labels list
EMOTION_LABELS = ['happy', 'nostalgia', 'negative', 'neutral', 'like', 'curious', 'grateful', 'agreement', 'surprise']
INTENT_LABELS = ['seeking_recommendation', 'feedback', 'chit_chat', 'other']

emotion_model_path = "/data/s3905993/ECRHMAS/src/models/roberta_emotion_classifier"
intent_model_path = "/data/s3905993/ECRHMAS/src/models/roberta_intent_classifier"

agent = EmotionIntentRecognizer(
    emotion_model_path=emotion_model_path,
    intent_model_path=intent_model_path,
    emotion_labels=EMOTION_LABELS,
    intent_labels=INTENT_LABELS,
    history_turns=2,
    debug=True,
    device="cpu"
)

<<<<<<< HEAD
#example dialogue
=======
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
dialogue = [
    "System: What kind of movies do you like?",
    "User: I love old movies, especially ones from my childhood.",
    "System: Oh, that's great! Any particular genre?",
    "User: Not really, just something that makes me feel nostalgic."
]

result = agent.process(dialogue)
print(result)
