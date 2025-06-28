import json
with open("/data/s3905993/ECRHMAS/data/llama_test.json") as f:
    first_line = f.readline()
    print(type(json.loads(first_line)))

