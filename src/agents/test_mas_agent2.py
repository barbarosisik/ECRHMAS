from mas_agent2 import KnowledgeAwareResponderMAS

movie_ids_path = "/data/s3905993/ECRHMAS/data/redial_gen/movie_ids_aligned.json"
movie_names_path = "/data/s3905993/ECRHMAS/data/redial_gen/movie_name.json"
movie_years_path = "/data/s3905993/ECRHMAS/data/redial_gen/movie_years.json"
movie_genres_path = "/data/s3905993/ECRHMAS/data/redial_gen/movie_genres_full.json"
movie_kb_path = "/data/s3905993/ECRHMAS/data/redial_gen/movie_knowledge_base.json"   # NEW

#instantiate agent
agent2 = KnowledgeAwareResponderMAS(
    movie_ids_path=movie_ids_path,
    movie_names_path=movie_names_path,
    movie_years_path=movie_years_path,
    movie_genres_path=movie_genres_path,
    movie_kb_path=movie_kb_path,    # NEW
    debug=True
)

#example dialogue
dialogue = [
    "System: Welcome! How can I help?",
    "User: I'm in the mood for a classic sci-fi movie.",
    "System: Do you like space adventures?",
    "User: Yes, something like Interstellar."
]

user_state = {
    "intent": "seeking_recommendation",
    "intent_score": {"seeking_recommendation": 0.99, "other": 0.01}
}
result = agent2.process(dialogue, user_state)
print(result)
