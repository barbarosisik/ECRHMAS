import sys, json

movie_id = int(sys.argv[1])
with open('/data/s3905993/ECRHMAS/data/redial_gen/entity2id.json') as f:
    e2id = json.load(f)

matches = [k for k, v in e2id.items() if v == movie_id]
print(f"URIs mapped to ID {movie_id}:")
for uri in matches:
    print(uri)