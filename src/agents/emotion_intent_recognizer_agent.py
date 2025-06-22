import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

class EmotionIntentRecognizer:
    def __init__(
        self,
        emotion_model_path,
        intent_model_path,
        emotion_labels,
        intent_labels,
        device=None,
        history_turns=2,
        debug=False
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_turns = history_turns
        self.debug = debug

        #loading emotion model & tokenizer
        self.emotion_tokenizer = RobertaTokenizer.from_pretrained(emotion_model_path)
        self.emotion_model = RobertaForSequenceClassification.from_pretrained(emotion_model_path)
        self.emotion_model.to(self.device)
        self.emotion_labels = emotion_labels

        #loading intent model & tokenizer
        self.intent_tokenizer = RobertaTokenizer.from_pretrained(intent_model_path)
        self.intent_model = RobertaForSequenceClassification.from_pretrained(intent_model_path)
        self.intent_model.to(self.device)
        self.intent_labels = intent_labels

    def process(self, dialogue_history):
        """
        Args:
            dialogue_history: List[str], where each element is a dialogue turn (alternating user/system)
        Returns:
            dict with keys: emotion, intent, emotion_score, intent_score
        """

        context = " ".join(dialogue_history[-self.history_turns:])

        # --- Emotion classification ---
        emo_inputs = self.emotion_tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=128)
        emo_inputs = {k: v.to(self.device) for k, v in emo_inputs.items()}
        with torch.no_grad():
            emo_logits = self.emotion_model(**emo_inputs).logits
            emo_probs = F.softmax(emo_logits, dim=1).squeeze(0)
        top_emo_idx = int(torch.argmax(emo_probs).cpu())
        top_emo = self.emotion_labels[top_emo_idx]
        emotion_score = {label: float(prob) for label, prob in zip(self.emotion_labels, emo_probs.cpu().numpy())}

        # --- Intent classification ---
        intent_inputs = self.intent_tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=128)
        intent_inputs = {k: v.to(self.device) for k, v in intent_inputs.items()}
        with torch.no_grad():
            intent_logits = self.intent_model(**intent_inputs).logits
            intent_probs = F.softmax(intent_logits, dim=1).squeeze(0)
        top_intent_idx = int(torch.argmax(intent_probs).cpu())
        top_intent = self.intent_labels[top_intent_idx]
        intent_score = {label: float(prob) for label, prob in zip(self.intent_labels, intent_probs.cpu().numpy())}

        if self.debug:
            print(f"\n--- [Agent 1 Debug] ---")
            print(f"Context used: {repr(context)}")
            print(f"Emotion scores: {emotion_score}")
            print(f"Intent scores: {intent_score}")
            print(f"Predicted Emotion: {top_emo}, Predicted Intent: {top_intent}")

        return {
            "emotion": top_emo,
            "emotion_score": emotion_score,
            "intent": top_intent,
            "intent_score": intent_score,
        }
