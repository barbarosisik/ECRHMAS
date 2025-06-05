import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class ContextualEmotionPredictor:
    def __init__(
        self,
        model_path,
        label_list,
        device=None,
        context_turns=3,
        debug=False
    ):
        """
        model_path: Path to the trained model checkpoint (directory containing config.json, pytorch_model.bin, etc.)
        label_list: List of emotion labels, in the order used in training
        device: 'cpu' or 'cuda'
        context_turns: Number of recent dialogue turns to use
        debug: Print debug info
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.context_turns = context_turns
        self.debug = debug
        self.label_list = label_list

        # Load tokenizer and model. Using the same model_path ensures the
        # tokenizer matches the fine-tuned model checkpoint rather than the
        # base DialoGPT tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if getattr(self.tokenizer, "pad_token", None) is None:
            # Some conversational models do not define a pad token. For
            # classification we fall back to the EOS token for padding.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, dialogue_history):
        """
        dialogue_history: list of strings (alternating user/system)
        Returns: dict: {'emotion': str, 'emotion_score': dict}
        """

        context = " ".join(dialogue_history[-self.context_turns:])

        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze(0)

        top_idx = int(torch.argmax(probs).cpu())
        top_emotion = self.label_list[top_idx]
        emotion_score = {label: float(probs[i]) for i, label in enumerate(self.label_list)}

        if self.debug:
            print(f"\n--- [ContextualEmotionPredictor Debug] ---")
            print(f"Context: {repr(context)}")
            print(f"Emotion Scores: {emotion_score}")
            print(f"Predicted Emotion: {top_emotion}")

        return {
            "emotion": top_emotion,
            "emotion_score": emotion_score,
        }
