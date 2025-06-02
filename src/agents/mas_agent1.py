import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

class EmotionIntentRecognizerMAS:
    def __init__(
        self,
        intent_model_path,
        intent_labels,
        device=None,
        history_turns=3,
        debug=False
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_turns = history_turns
        self.debug = debug
        self.intent_labels = intent_labels

        # Load intent classifier and tokenizer
        self.intent_tokenizer = RobertaTokenizer.from_pretrained(intent_model_path)
        self.intent_model = RobertaForSequenceClassification.from_pretrained(intent_model_path)
        self.intent_model.to(self.device)
        self.intent_model.eval()

    def process(self, dialogue_history, gold_emotion_label):
        """
        Args:
            dialogue_history: List[str], each a dialogue turn (alternating user/system)
            gold_emotion_label: The true emotion label (from dataset)
        Returns:
            dict: {'emotion': str, 'intent': str, 'intent_score': dict}
        """
        
        context = " ".join(dialogue_history[-self.history_turns:])

        # --- Intent classification ---
        inputs = self.intent_tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.intent_model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze(0)
        top_idx = int(torch.argmax(probs).cpu())
        top_intent = self.intent_labels[top_idx]
        intent_score = {label: float(probs[i]) for i, label in enumerate(self.intent_labels)}

        if self.debug:
            print("\n--- [Agent 1 Debug] ---")
            print(f"Context: {repr(context)}")
            print(f"Intent Scores: {intent_score}")
            print(f"Predicted Intent: {top_intent}")
            print(f"Gold Emotion: {gold_emotion_label}")

        return {
            "emotion": gold_emotion_label,
            "intent": top_intent,
            "intent_score": intent_score,
        }
