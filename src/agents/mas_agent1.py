import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json
import os

class IntentEmotionRecognizerMAS:
    def __init__(
        self,
        intent_model_path,
        intent_label_list,
        history_turns=3,
        device=None,
        debug=False
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_turns = history_turns
        self.debug = debug

        # Load intent classifier
        self.intent_tokenizer = RobertaTokenizer.from_pretrained(intent_model_path)
        self.intent_model = RobertaForSequenceClassification.from_pretrained(intent_model_path)
        self.intent_model.to(self.device)
        self.intent_model.eval()
        self.intent_labels = intent_label_list

    def process(self, dialogue_history, gold_emotion_label):
        """
        Args:
            dialogue_history: List[str], the dialogue turns (user/system)
            gold_emotion_label: str, gold emotion from dataset
        Returns:
            dict: {'emotion': ..., 'emotion_score': ..., 'intent': ..., 'intent_score': ...}
        """
        context = " ".join(dialogue_history[-self.history_turns:])

        # Intent prediction (RoBERTa)
        intent_inputs = self.intent_tokenizer(
            context, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        intent_inputs = {k: v.to(self.device) for k, v in intent_inputs.items()}
        with torch.no_grad():
            intent_logits = self.intent_model(**intent_inputs).logits
            intent_probs = F.softmax(intent_logits, dim=1).squeeze(0)
        top_intent_idx = int(torch.argmax(intent_probs).cpu())
        top_intent = self.intent_labels[top_intent_idx]
        intent_score = {label: float(intent_probs[i]) for i, label in enumerate(self.intent_labels)}

        if self.debug:
            print("\n--- [MAS Agent 1 Debug] ---")
            print(f"Context used: {repr(context)}")
            print(f"Gold Emotion: {gold_emotion_label}")
            print(f"Intent scores: {intent_score}")
            print(f"Predicted Intent: {top_intent}")

        return {
            "emotion": gold_emotion_label,              # Direct from dataset!
            "emotion_score": {gold_emotion_label: 1},   # 1-hot
            "intent": top_intent,
            "intent_score": intent_score,
        }
