import json
import requests
import time

#api keys I used to extract
API_KEYS = [
    "c23d62e3",
    "4ecbdf4c",
    "83b612d7",
    "fff21664",
    "385416ad"
]
BATCH_SIZE = 1000  #keys limited to 1000 daily.

MOVIE_NAME_PATH = "/data/s3905993/ECRHMAS/data/redial_gen/movie_name.json"
MOVIE_YEAR_PATH = "/data/s3905993/ECRHMAS/data/redial_gen/movie_years.json"
OUT_BASE = "/data/s3905993/ECRHMAS/data/redial_gen/movie_genres_part"

with open(MOVIE_NAME_PATH, "r") as f:
    movie_names = json.load(f)
with open(MOVIE_YEAR_PATH, "r") as f:
    movie_years = json.load(f)

assert len(movie_names) == len(movie_years)
total_movies = len(movie_names)
num_batches = (total_movies + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in range(num_batches):
    api_key = API_KEYS[batch_num]
    start = batch_num * BATCH_SIZE
    end = min((batch_num + 1) * BATCH_SIZE, total_movies)
    genres_all = []

    print(f"\nStarting batch {batch_num+1}/{num_batches} with API key: {api_key}")
    for i in range(start, end):
        name, year = movie_names[i], movie_years[i]
        params = {
            "t": name.split(' (')[0],
            "y": year if year else "",
            "apikey": api_key,
            "type": "movie"
        }
        r = requests.get("http://www.omdbapi.com/", params=params)
        genres = []
        try:
            data = r.json()
            if "Genre" in data and data["Genre"]:
                genres = [g.strip().lower() for g in data["Genre"].split(",")]
        except Exception as e:
            print(f"Error fetching genre for: {name} - {e}")
        genres_all.append(genres)
        print(f"{i+1}/{total_movies}: {name} ({year}) -> {genres}")
        time.sleep(0.2)

    out_file = f"{OUT_BASE}{batch_num+1}.json"
    with open(out_file, "w") as f:
        json.dump(genres_all, f)
    print(f"Batch {batch_num+1} done: {out_file}")

print("All batches complete!")
