import json
import os
import random
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

class KnowledgeAwareResponderMAS:
    def __init__(
        self,
        movie_ids_path: str,
        movie_names_path: str,
        movie_years_path: str = None,
        movie_genres_path: str = None,
        movie_kb_path: str = None,
        dialogpt_model_path: str = None,
        emotion_model_path: str = "SamLowe/roberta-base-go_emotions",
        debug: bool = False
    ):
        self.debug = debug
        self.device = torch.device("cpu")
        #self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
        #loading emotion classifier
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path).to(self.device)
        #loading dialogpt model
        if dialogpt_model_path is not None:
            self.dialogpt_tokenizer = AutoTokenizer.from_pretrained(dialogpt_model_path)
            self.dialogpt_model = AutoModelForCausalLM.from_pretrained(dialogpt_model_path).to(self.device)
        else:
            self.dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
            self.dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large").to(self.device)

<<<<<<< HEAD
        #loading movie ID - name mapping
=======
        #loading movie ID <-> name mapping (lists, aligned by index)
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
        with open(movie_ids_path, "r") as f:
            self.movie_ids = json.load(f)
        with open(movie_names_path, "r") as f:
            self.movie_names = json.load(f)

        self.movie_years = None
        self.movie_genres = None
        if movie_years_path is not None and os.path.exists(movie_years_path):
            with open(movie_years_path, "r") as f:
                self.movie_years = json.load(f)
        if movie_genres_path is not None and os.path.exists(movie_genres_path):
            with open(movie_genres_path, "r") as f:
                self.movie_genres = json.load(f)

        #loading Knowledge Base
        self.movie_kb = {}
        if movie_kb_path and os.path.exists(movie_kb_path):
            with open(movie_kb_path, "r") as f:
                self.movie_kb = json.load(f)

        self.EMOTION_TO_GENRES = {
            "nostalgia": ["drama", "family", "romance", "adventure", "music", "history", "biography", "fantasy"],
            "happy": ["comedy", "animation", "musical", "adventure", "family", "fantasy", "sport", "music"],
            "sad": ["drama", "history", "biography"],
            "angry": ["action", "crime", "thriller", "war"],
            "scared": ["horror", "thriller", "mystery"],
            "curious": ["mystery", "sci-fi", "adventure", "documentary", "film-noir", "history", "reality-tv", "news"],
            "bored": ["short", "animation", "comedy", "adventure"],
            "other": ["action", "adult", "adventure", "animation", "biography", "comedy", "crime", "documentary", "drama", "family", "fantasy", "film-noir", "history", "horror", "music", "musical", "mystery", "n/a", "news", "reality-tv", "romance", "sci-fi", "short", "sport", "talk-show", "thriller", "war", "western"]
        }

    def predict_emotion(self, text: str) -> str:
        inputs = self.emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.emotion_model(**inputs).logits
        pred_idx = logits.argmax(-1).item()
        emotion = self.emotion_model.config.id2label[pred_idx].lower()
        return emotion

    def extract_mentioned_movies(self, dialogue_context: List[str]) -> List[str]:
        mentioned = set()
        for utter in dialogue_context:
            for name in self.movie_names:
                if name.lower() in utter.lower():
                    mentioned.add(name)
        return list(mentioned)

    def filter_candidates(self, user_state: Dict[str, Any], mentioned: List[str]) -> List[int]:
        emotion = user_state.get("emotion", "other").lower()
        filter_indices = list(range(len(self.movie_names)))

        if emotion == "nostalgia" and self.movie_years:
            filter_indices = [i for i in filter_indices if self.movie_years[i] and self.movie_years[i] < 2000]
        elif self.movie_genres and emotion in self.EMOTION_TO_GENRES:
            preferred_genres = set(self.EMOTION_TO_GENRES[emotion])
            filter_indices = [i for i in filter_indices if self.movie_genres[i] and any(g.lower() in preferred_genres for g in self.movie_genres[i])]

        if mentioned:
            filter_indices = [i for i in filter_indices if self.movie_names[i] not in mentioned]
        return filter_indices

    def retrieve_knowledge(self, movie_name: str) -> Dict[str, Any]:
        return self.movie_kb.get(movie_name, {})

    def rank_candidates(self, dialogue_context: List[str], candidate_indices: List[int]) -> int:
        if not candidate_indices:
            return None
        return random.choice(candidate_indices)
    
<<<<<<< HEAD
    def generate_dialogpt_response(self, context: list, knowledge: str = "", user_emotion: str = "", max_new_tokens: int = 80) -> str:
        prompt = f"User feels {user_emotion}.\n"  #injecting user emotion
        prompt += "\n".join(context)
        if knowledge:
            prompt += f"\n[KNOWLEDGE]: {knowledge}"

        #Debug prints for knowledge integration
        if self.debug:
            print("\n[DEBUG] FINAL PROMPT SENT TO DialogGPT:\n", prompt)
            print("[DEBUG] TOKENIZED LENGTH:", len(self.dialogpt_tokenizer(prompt).input_ids))

=======
    def generate_dialogpt_response(self, context: list, knowledge: str = "", max_new_tokens: int = 60) -> str:
        history = "\n".join(context)
        prompt = history
        if knowledge:
            prompt += f"\n[KNOWLEDGE]: {knowledge}"

>>>>>>> fa533849599714b8f490bc557652c007aa45e497
        #tokenize and generate
        inputs = self.dialogpt_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print("device:", next(self.dialogpt_model.parameters()).device)
        print("inputs['input_ids'].device:", inputs["input_ids"].device)
        print("inputs['input_ids'].shape:", inputs["input_ids"].shape)
        with torch.no_grad():
            outputs = self.dialogpt_model.generate(
                inputs["input_ids"],
                max_new_tokens=80,
                #max_new_tokens=max_new_tokens,
                pad_token_id=self.dialogpt_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                early_stopping=True,
                top_k=50
            )
        response = self.dialogpt_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response.strip()
<<<<<<< HEAD
    
    def generate_response(self, recommended_movie, knowledge_used, dialogue_context, user_emotion=""):
    #def generate_response(self, recommended_movie: Dict[str, Any], knowledge_used: Dict[str, Any], dialogue_context: list) -> str:
        review = ""
        if knowledge_used and "reviews" in knowledge_used and knowledge_used["reviews"]:
            #concatenate first review title and first 1-2 sentences
=======

    def generate_response(self, recommended_movie: Dict[str, Any], knowledge_used: Dict[str, Any], dialogue_context: list) -> str:
        review = ""
        if knowledge_used and "reviews" in knowledge_used and knowledge_used["reviews"]:
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
            review_obj = knowledge_used["reviews"][0]
            review = review_obj.get("title", "")
            if review_obj.get("content"):
                review += ": " + " ".join(review_obj["content"][:1])

<<<<<<< HEAD
=======
        #optionally, including movie name/director in knowledge
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
        movie_facts = ""
        if recommended_movie:
            movie_facts += f"{recommended_movie['movie_name']} ({recommended_movie.get('year', '')}), genres: {', '.join(recommended_movie.get('genres', []) or [])}."
        knowledge_for_prompt = f"{movie_facts} {review}".strip()

<<<<<<< HEAD
        #using DialogGPT to generate the actual response
        return self.generate_dialogpt_response(
            dialogue_context,
            knowledge=knowledge_for_prompt,
            user_emotion=user_emotion)

    def process(self, dialogue_context, user_state):
    #def process(self, dialogue_context: List[str], user_state: Dict[str, Any]) -> Dict[str, Any]:
        #predicting emotion using the last user utterance (assuming user is always last speaker)
=======
        #DialogGPT to generate the actual response
        return self.generate_dialogpt_response(dialogue_context, knowledge=knowledge_for_prompt)

    def process(self, dialogue_context: List[str], user_state: Dict[str, Any]) -> Dict[str, Any]:
        #predicting emotion using the last user utterance (assume user is always last speaker)
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
        if dialogue_context:
            user_utterances = [utt for utt in dialogue_context if utt.lower().startswith("user:")]
            if user_utterances:
                last_user_utterance = user_utterances[-1].replace("User:", "").strip()
                predicted_emotion = self.predict_emotion(last_user_utterance)
                user_state["emotion"] = predicted_emotion
                if self.debug:
                    print(f"[DEBUG] Detected user emotion: {predicted_emotion}")
                    print("[DEBUG] user_state['emotion'] before generate_response:", user_state.get("emotion"))
        mentioned = self.extract_mentioned_movies(dialogue_context)
        candidate_indices = self.filter_candidates(user_state, mentioned)
        chosen_idx = self.rank_candidates(dialogue_context, candidate_indices)

        recommended_items = []
        recommended_movie = None
        knowledge_used = {}

        if chosen_idx is not None:
            recommended_movie = {"movie_id": self.movie_ids[chosen_idx],
                                 "movie_name": self.movie_names[chosen_idx],
                                 "year": self.movie_years[chosen_idx] if self.movie_years else None,
                                 "genres": self.movie_genres[chosen_idx] if self.movie_genres else None}

            knowledge_used = self.retrieve_knowledge(self.movie_names[chosen_idx])
            recommended_items.append(recommended_movie)

        response = self.generate_response(
            recommended_movie,
            knowledge_used,
            dialogue_context,
            user_state.get("emotion", "")
        )

        agent_logs = {}
        if self.debug:
            agent_logs["debug_info"] = {"input_dialogue": dialogue_context,
                                        "user_state": user_state,
                                        "mentioned": mentioned,
                                        "candidate_indices": candidate_indices,
                                        "chosen_movie": recommended_movie,
                                        "knowledge_used": knowledge_used}

        return {"response": response,
                "recommended_items": recommended_items,
                "knowledge_used": knowledge_used,
                "agent_logs": agent_logs}
