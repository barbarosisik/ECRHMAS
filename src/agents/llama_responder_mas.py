import random
from .llama_response_agent import LlamaResponseAgent

class LlamaResponderMAS:
    def __init__(self, model_dir, device="cuda", debug=False):
        self.agent = LlamaResponseAgent(model_dir, device=device, debug=debug)
        self.debug = debug
        #fallback templates for emotion alignment
        self.fallback_templates = [
            "I recommend {movie} because it matches your current feelings.",
            "{movie} would be a wonderful pick given your mood.",
            "Considering how you're feeling, {movie} is a great choice.",
            "If you want something that fits your emotion, try {movie}.",
            "{movie} is a movie I think you'll appreciate right now.",
            "Since you're feeling {emotion}, {movie} could be a perfect fit.",
            "You might enjoy {movie}; it suits your mood.",
            "How about watching {movie}? It could be just right for you.",
            "I suggest {movie} for your current mood."
        ]

    def build_prompt(self, dialogue_context, user_emotion, knowledge="", movie_name="a movie"):
        """Construct the Llama2 system prompt for a single user dialog context."""
        system_msg = (
            "You are an empathetic movie assistant. "
            "Always mention the movie title and adapt your tone to the user's emotion. "
            "Be concise and empathetic."
        )
        context_block = ""
        for i, utter in enumerate(dialogue_context):
            speaker = "User" if i % 2 == 0 else "Assistant"
            context_block += f"{speaker}: {utter}\n"
        if knowledge:
            knowledge_block = f"\n[KNOWLEDGE]\n{knowledge}\n"
        else:
            knowledge_block = ""
        prompt = (
            f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n"
            f"<USER_EMOTION> {user_emotion} </USER_EMOTION>\n"
            f"{context_block.strip()}{knowledge_block}\nAssistant:"
        )
        return prompt

    def generate_response(self, dialogue_context, user_emotion, knowledge="", movie_name="a movie", max_new_tokens=120):
        prompt = self.build_prompt(dialogue_context, user_emotion, knowledge, movie_name)
        response = self.agent.generate(prompt, max_new_tokens=max_new_tokens)
        #fallback if movie not present
        movie_name = str(movie_name)
        if movie_name.lower() not in response.lower():
            emotion = user_emotion if user_emotion else "your current mood"
            template = random.choice(self.fallback_templates)
            response = template.format(movie=movie_name, emotion=emotion)
        return response
