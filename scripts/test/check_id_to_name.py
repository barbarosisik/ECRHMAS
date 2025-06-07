import sys, json

movie_id = int(sys.argv[1])
with open('/data/s3905993/ECRHMAS/data/redial_gen/movie_ids.json') as f:
    mids = json.load(f)
with open('/data/s3905993/ECRHMAS/data/redial_gen/movie_name.json') as f:
    mnames = json.load(f)

if movie_id in mids:
    idx = mids.index(movie_id)
    print(f"ID {movie_id} --> Name: {mnames[idx]}")
else:
    print("ID not found in movie_ids.json")
