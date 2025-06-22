import argparse
import os
import json
from tqdm.auto import tqdm

from mas_agent1 import IntentEmotionRecognizerMAS
from mas_agent2 import KnowledgeAwareResponderMAS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data JSONL (one conversation per line)")
    parser.add_argument("--output_file", type=str, default="results/ecrhmas_eval.jsonl",
                        help="Where to save output results")
    #agent 1
    parser.add_argument("--intent_model_path", type=str, required=True)
    parser.add_argument("--intent_label_list", type=str, default=None, help="Path to intent label list JSON")
    parser.add_argument("--emotion_model_path", type=str, required=True)
    parser.add_argument("--emotion_label_list", type=str, default=None, help="Path to emotion label list TXT (one label per line)")
    #agent 2
    parser.add_argument("--movie_ids_path", type=str, required=True)
    parser.add_argument("--movie_names_path", type=str, required=True)
    parser.add_argument("--movie_years_path", type=str)
    parser.add_argument("--movie_genres_path", type=str)
    parser.add_argument("--movie_kb_path", type=str)
    parser.add_argument("--dialogpt_model_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    #- loading test data -
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f]
    if args.max_test_samples:
        test_samples = test_samples[:args.max_test_samples]
    print(f"[INFO] Loaded {len(test_samples)} test samples from {args.test_data}")

    #- loading intent label list -
    if args.intent_label_list:
        with open(args.intent_label_list, "r") as f:
            intent_label_list = json.load(f)
    else:
        with open(os.path.join(args.intent_model_path, "intent_labels.json"), "r") as f:
            intent_label_list = json.load(f)

    #loading emotion label list
    if args.emotion_label_list:
        with open(args.emotion_label_list, "r") as f:
            emotion_label_list = [line.strip() for line in f if line.strip()]
    else:
        label_names_path = os.path.join(args.emotion_model_path, "label_names.txt")
        with open(label_names_path, "r") as f:
            emotion_label_list = [line.strip() for line in f if line.strip()]

    #- agents -
    agent1 = IntentEmotionRecognizerMAS(
        intent_model_path=args.intent_model_path,
        intent_label_list=intent_label_list,
        emotion_model_path=args.emotion_model_path,
        emotion_label_list=emotion_label_list,
        debug=args.debug
    )
    agent2 = KnowledgeAwareResponderMAS(
        movie_ids_path=args.movie_ids_path,
        movie_names_path=args.movie_names_path,
        movie_years_path=args.movie_years_path,
        movie_genres_path=args.movie_genres_path,
        movie_kb_path=args.movie_kb_path,
        dialogpt_model_path=args.dialogpt_model_path,
        debug=args.debug
    )

    results = []

    for sample in tqdm(test_samples, desc="Running ECRHMAS Inference"):
<<<<<<< HEAD
        #preparing context
        context = sample["context"]  #list of dialogue utterances
=======
        # --- Prepare context ---
        context = sample["context"]
>>>>>>> fa533849599714b8f490bc557652c007aa45e497

        #Agent 1: Recognize intent + emotion
        agent1_result = agent1.process(context)

        #Agent 2: Generate recommendation + response
        agent2_result = agent2.process(
            context,
<<<<<<< HEAD
            agent1_result  #user_state: 'intent', 'intent_score', 'emotion', 'emotion_score'
=======
            agent1_result
>>>>>>> fa533849599714b8f490bc557652c007aa45e497
        )

        # result
        entry = {
            "context": context,
            "user_state": agent1_result,
            "recommended_items": agent2_result.get("recommended_items"),
            "knowledge_used": agent2_result.get("knowledge_used"),
            "response": agent2_result.get("response"),
            "ground_truth_response": sample.get("resp", ""),
            "ground_truth_rec": sample.get("rec", None),
            "agent_logs": agent2_result.get("agent_logs", {})
        }
        results.append(entry)

    #save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(results)} outputs to: {args.output_file}")

if __name__ == "__main__":
    main()
