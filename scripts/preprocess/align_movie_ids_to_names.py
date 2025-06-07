# align_movie_ids_to_names.py
import json

name_path = "/data/s3905993/ECRHMAS/data/redial_gen/movie_name.json"
id_path = "/data/s3905993/ECRHMAS/data/redial_gen/movie_ids.json"
out_path = "/data/s3905993/ECRHMAS/data/redial_gen/movie_ids_aligned.json"

with open(name_path, "r") as f:
    names = json.load(f)

with open(id_path, "r") as f:
    ids = json.load(f)

# If original id file is a mapping, you'll need a mapping from name to id (or vice versa).
# But if it's just a list and the first 4989 are correct, just truncate:
ids_aligned = ids[:len(names)]

with open(out_path, "w") as f:
    json.dump(ids_aligned, f)

print(f"Trimmed ids to match names: {len(ids_aligned)}")
