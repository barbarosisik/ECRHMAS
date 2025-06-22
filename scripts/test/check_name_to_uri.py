import sys, json

movie_name = sys.argv[1]
with open('/data/s3905993/ECRHMAS/data/redial_gen/movie_name.json') as f:
    mnames = json.load(f)
with open('/data/s3905993/ECRHMAS/data/redial_gen/movie_ids.json') as f:
    mids = json.load(f)
with open('/data/s3905993/ECRHMAS/data/redial_gen/entity2id.json') as f:
    e2id = json.load(f)

idxs = [i for i, name in enumerate(mnames) if name == movie_name]
for i in idxs:
    movie_id = mids[i]
    uris = [k for k, v in e2id.items() if v == movie_id]
    print(f"Name: {movie_name}, idx: {i}, ID: {movie_id}")
    print(f"URIs: {uris}")