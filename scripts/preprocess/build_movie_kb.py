import json
import os
from tqdm import tqdm
from dataset_dbpedia import DBpedia

# Paths (modify as needed)
MOVIE_NAMES_PATH = "data/redial_gen/movie_name.json"
REVIEWS_PATH = "data/redial_gen/movie_reviews_filted_0.1_confi.json"
OUT_PATH = "data/redial_gen/movie_knowledge_base.json"

# Initialize DBpedia instance (dataset="redial_gen")
dbpedia = DBpedia(dataset="redial_gen")

# Load movie names
with open(MOVIE_NAMES_PATH, "r") as f:
    movie_names = json.load(f)

# Load IMDb reviews (filtered)
with open(REVIEWS_PATH, "r", encoding='utf-8') as f:
    movie_reviews = json.load(f)

knowledge_base = {}

for name in tqdm(movie_names, desc="Building Knowledge Base"):
    movie_entry = {}
    
    # Extract DBpedia 1-hop neighborhood
    clean_name = dbpedia.entity2name(name, flag=True)
    try:
        neighborhood = dbpedia.get_one_hop_neighborhood(clean_name, flag=False)
    except:
        neighborhood = {}

    entities = [dbpedia.entity2name(e, flag=True) for e in neighborhood.keys()]

    # Prepare movie entry
    movie_entry['description'] = f"A film titled '{name}'. More details available via entities."
    movie_entry['entities'] = entities

    # Add top reviews (max 3)
    reviews = []
    if name in movie_reviews:
        for review in movie_reviews[name][:3]:  # Top 3 reviews
            review_entry = {
                "title": review.get("title", ""),
                "content": review.get("content", []),
                "entities": [dbpedia.entity2name(e[0], flag=True) for e in review.get("content_e_0.1", [])[:5]]  # Top 5 entities
            }
            reviews.append(review_entry)

    movie_entry['reviews'] = reviews

    knowledge_base[name] = movie_entry

# Save knowledge base to JSON
with open(OUT_PATH, "w", encoding='utf-8') as f:
    json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

print(f"Knowledge base created successfully at {OUT_PATH}")
