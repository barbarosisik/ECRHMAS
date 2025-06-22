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
        emotion_model_path, 
        emotion_label_list, 
        history_turns=3,
        device=None,
        debug=False
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_turns = history_turns
        self.debug = debug

<<<<<<< HEAD
        #intent classifier
=======
        #loading intent classifier
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
        self.intent_tokenizer = RobertaTokenizer.from_pretrained(intent_model_path)
        self.intent_model = RobertaForSequenceClassification.from_pretrained(intent_model_path)
        self.intent_model.to(self.device)
        self.intent_model.eval()
        self.intent_labels = intent_label_list

        #emotion classifier
        self.emotion_tokenizer = RobertaTokenizer.from_pretrained(emotion_model_path)
        self.emotion_model = RobertaForSequenceClassification.from_pretrained(emotion_model_path)
        self.emotion_model.to(self.device)
        self.emotion_model.eval()
        self.emotion_labels = emotion_label_list

    def process(self, dialogue_history, _gold_emotion_label=None):
        """
        Args:
            dialogue_history: List[str], the dialogue turns (user/system)
        Returns:
            dict: {'emotion': ..., 'emotion_score': ..., 'intent': ..., 'intent_score': ...}
        """
        #intent(using last N turns as context)
        context = " ".join(dialogue_history[-self.history_turns:])
<<<<<<< HEAD
=======

        #Intent prediction (RoBERTa)
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
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

        #emotion(uses last user utterance only)
        user_utterances = [utt for utt in dialogue_history if utt.lower().startswith("user:")]
        if user_utterances:
            last_user_utterance = user_utterances[-1].replace("User:", "").strip()
        else:
            last_user_utterance = dialogue_history[-1] if dialogue_history else ""
        emotion_inputs = self.emotion_tokenizer(
            last_user_utterance, return_tensors="pt", truncation=True, padding=True, max_length=64
        )
        emotion_inputs = {k: v.to(self.device) for k, v in emotion_inputs.items()}
        with torch.no_grad():
            emotion_logits = self.emotion_model(**emotion_inputs).logits
            emotion_probs = F.softmax(emotion_logits, dim=1).squeeze(0)
        top_emotion_idx = int(torch.argmax(emotion_probs).cpu())
        top_emotion = self.emotion_labels[top_emotion_idx]
        emotion_score = {label: float(emotion_probs[i]) for i, label in enumerate(self.emotion_labels)}

        if self.debug:
            print("\n--- [MAS Agent 1 Debug] ---")
            print(f"Context used: {repr(context)}")
            print(f"Last user utterance: {repr(last_user_utterance)}")
            print(f"Predicted Emotion: {top_emotion}")
            print(f"Predicted Intent: {top_intent}")
            print(f"Intent scores: {intent_score}")
            print(f"Emotion scores: {emotion_score}")

        return {
<<<<<<< HEAD
            "emotion": top_emotion,
            "emotion_score": emotion_score,
=======
            "emotion": gold_emotion_label,
            "emotion_score": {gold_emotion_label: 1},
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
            "intent": top_intent,
            "intent_score": intent_score,
        }
