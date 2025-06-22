import json
import re

MOVIE_NAME_PATH = "/data/s3905993/ECRHMAS/data/redial_gen/movie_name.json"
OUT_PATH = "/data/s3905993/ECRHMAS/data/redial_gen/movie_years.json"

with open(MOVIE_NAME_PATH, "r") as f:
    movie_names = json.load(f)

movie_years = []
year_pattern = re.compile(r"\((\d{4})\)")

for name in movie_names:
    match = year_pattern.search(name)
    if match:
        movie_years.append(int(match.group(1)))
    else:
        #using 0 or None if year missing
        movie_years.append(None)

with open(OUT_PATH, "w") as f:
    json.dump(movie_years, f)

print(f"Extracted years for {len(movie_names)} movies. Saved to: {OUT_PATH}")
