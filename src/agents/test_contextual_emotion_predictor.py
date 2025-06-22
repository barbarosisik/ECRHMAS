from contextual_emotion_predictor_agent import ContextualEmotionPredictor

#model checkpoint directory
MODEL_PATH = "/data/s3905993/ECRMAS/ECR-main/src_emo/data/saved/emp_conv2025-04-01-10-57-46/"
LABEL_LIST = ['happy', 'nostalgia', 'negative', 'neutral', 'like', 'curious', 'grateful', 'agreement', 'surprise']

predictor = ContextualEmotionPredictor(
    model_path=MODEL_PATH,
    label_list=LABEL_LIST,
    context_turns=3,   #using last 3 turns
    debug=True
)

#example test dialogue
dialogue = [
    "System: What kind of movies do you like?",
    "User: I love old movies, especially ones from my childhood.",
    "System: Oh, that's great! Any particular genre?",
    "User: Not really, just something that makes me feel nostalgic."
]

result = predictor.predict(dialogue)
print(result)
