import json

def extract_field(lines, prefix):
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""

def convert_file(llama_input, output_jsonl):
    with open(llama_input, "r") as f:
        llama_samples = json.load(f)
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for entry in llama_samples:
            lines = entry["input"].split('\n')
            movie_name = extract_field(lines, "Movie Name:")
            first_sentence = extract_field(lines, "First Sentence:")
            context = [f"User: {first_sentence}"] if first_sentence else []
            rec = [movie_name] if movie_name else []
            fout.write(json.dumps({
                "context": context,
                "resp": entry.get("output", "").strip(),
                "rec": rec,
                "knowledge_used": entry["input"],
            }) + "\n")
    print(f"Converted {len(llama_samples)} samples to ECRMAS format: {output_jsonl}")

llama_test_input = "/data/s3905993/ECRHMAS/data/llama_test.json"
llama_train_input = "/data/s3905993/ECRHMAS/data/llama_train.json"
llama_test_output = "/data/s3905993/ECRHMAS/data/llama_test_converted.jsonl"
llama_train_output = "/data/s3905993/ECRHMAS/data/llama_train_converted.jsonl"

convert_file(llama_train_input, llama_train_output)
convert_file(llama_test_input, llama_test_output)
