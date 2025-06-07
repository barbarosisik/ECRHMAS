import json

PARTS = [
    "/data/s3905993/ECRHMAS/data/redial_gen/movie_genres_part1.json",
    "/data/s3905993/ECRHMAS/data/redial_gen/movie_genres_part2.json",
    "/data/s3905993/ECRHMAS/data/redial_gen/movie_genres_part3.json",
    "/data/s3905993/ECRHMAS/data/redial_gen/movie_genres_part4.json",
    "/data/s3905993/ECRHMAS/data/redial_gen/movie_genres_part5.json"
]
OUT_PATH = "/data/s3905993/ECRHMAS/data/redial_gen/movie_genres_full.json"

all_genres = []
for part in PARTS:
    with open(part, "r") as f:
        all_genres.extend(json.load(f))

with open(OUT_PATH, "w") as f:
    json.dump(all_genres, f)

print(f"Concatenated all {len(PARTS)} files into {OUT_PATH}")
print(f"Total movies: {len(all_genres)}")
