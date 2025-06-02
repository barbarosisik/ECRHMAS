from mas_agent1 import EmotionIntentRecognizerMAS

INTENT_LABELS = ['seeking_recommendation', 'feedback', 'chit_chat', 'other']

intent_model_path = "/data/s3905993/ECRMAS/models/intent_classifier"

agent = EmotionIntentRecognizerMAS(
    intent_model_path=intent_model_path,
    intent_labels=INTENT_LABELS,
    history_turns=3,
    debug=True
)

dialogue = [
    "System: What kind of movies do you like?",
    "User: I love old movies, especially ones from my childhood.",
    "System: Oh, that's great! Any particular genre?",
    "User: Not really, just something that makes me feel nostalgic."
]
gold_emotion_label = "nostalgia"

# Run the agent
result = agent.process(dialogue, gold_emotion_label)
print(result)
