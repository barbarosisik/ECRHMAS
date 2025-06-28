## MY KNOWLEDGE BASE LOGIC ##
import argparse
import os
import json
from tqdm.auto import tqdm

from mas_agent1 import IntentEmotionRecognizerMAS
from mas_agent2 import KnowledgeAwareResponderMAS
from agents.llama_responder_mas import LlamaResponderMAS

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
    #agent 2 (DialoGPT agent)
    parser.add_argument("--movie_ids_path", type=str, required=True)
    parser.add_argument("--movie_names_path", type=str, required=True)
    parser.add_argument("--movie_years_path", type=str)
    parser.add_argument("--movie_genres_path", type=str)
    parser.add_argument("--movie_kb_path", type=str)
    parser.add_argument("--dialogpt_model_path", type=str, required=True)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_test_samples", type=int, default=None)

    #llama2-Chat
    parser.add_argument("--llama_model_dir", type=str, default=None,
                        help="Path to Llama-2-Chat HF folder for LlamaResponderMAS")
    parser.add_argument("--llama_device", type=str, default="cpu")
    return parser.parse_args()

def build_short_kb(kb_entry, max_entities=2, max_review_sents=2):
    """
    Returning a concise KB string containing description, up to max_entities, and up to max_review_sents sentences from the first review.
    """
    kb_lines = []
    if isinstance(kb_entry, dict):
        #description
        desc = kb_entry.get('description')
        if desc:
            kb_lines.append(f"description: {desc}")

        #entities
        entities = kb_entry.get('entities', [])
        if entities:
            short_entities = ', '.join(entities[:max_entities])
            kb_lines.append(f"entities: {short_entities}")

        #reviews
        reviews = kb_entry.get('reviews') or kb_entry.get('review')
        if reviews:
            if isinstance(reviews, list):
                review = reviews[0] if reviews else None
                if isinstance(review, dict):
                    review_text = ' '.join(review.get('content', []) if isinstance(review.get('content', []), list) else [review.get('content', "")])
                else:
                    review_text = str(review) if review else ""
            elif isinstance(reviews, str):
                review_text = reviews
            else:
                review_text = ""
            #splitting into sentences and keeping the first N
            review_sentences = review_text.split('.')
            review_short = '. '.join([s.strip() for s in review_sentences if s.strip()][:max_review_sents])
            if review_short:
                kb_lines.append(f"review: {review_short}.")
    return '\n'.join(kb_lines)

def main():
    args = parse_args()

    #loading test data
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]
    if args.max_test_samples:
        test_samples = test_samples[:args.max_test_samples]
    print(f"[INFO] Loaded {len(test_samples)} test samples from {args.test_data}")

    #loading new movie kb
    if args.movie_kb_path and os.path.exists(args.movie_kb_path):
        with open(args.movie_kb_path, "r") as f:
            movie_kb = json.load(f)
    else:
        movie_kb = {}

    #loading intent label list
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

    #load movie ID-name mapping (for only once)
    with open(args.movie_ids_path, "r") as f:
        movie_ids = json.load(f)
    with open(args.movie_names_path, "r") as f:
        movie_names = json.load(f)

    #agent1 instantiation
    agent1 = IntentEmotionRecognizerMAS(
        intent_model_path=args.intent_model_path,
        intent_label_list=intent_label_list,
        emotion_model_path=args.emotion_model_path,
        emotion_label_list=emotion_label_list,
        debug=args.debug
    )

    #agent2 instantiation (Llama or DialogGPT)
    if args.llama_model_dir:
        print("[INFO] Using LlamaResponderMAS agent for response generation.")
        agent2 = LlamaResponderMAS(
            model_dir=args.llama_model_dir,
            device=args.llama_device,
            debug=args.debug
        )
        llama_mode = True
    else:
        print("[INFO] Using KnowledgeAwareResponderMAS (DialoGPT) agent.")
        agent2 = KnowledgeAwareResponderMAS(
            movie_ids_path=args.movie_ids_path,
            movie_names_path=args.movie_names_path,
            movie_years_path=args.movie_years_path,
            movie_genres_path=args.movie_genres_path,
            movie_kb_path=args.movie_kb_path,
            dialogpt_model_path=args.dialogpt_model_path,
            debug=args.debug
        )
        llama_mode = False

    results = []

    for sample in tqdm(test_samples, desc="Running ECRHMAS Inference"):
        context = sample["context"]

        #agent 1: Recognize intent + emotion
        agent1_result = agent1.process(context)

        if llama_mode:
            #movie name extraction
            movie_name = None
            rec_items = sample.get("recommended_items") or sample.get("rec") or []
            if isinstance(rec_items, list) and len(rec_items) > 0:
                first_item = rec_items[0]
                if isinstance(first_item, int):
                    #map movie id to name
                    if first_item in movie_ids:
                        idx = movie_ids.index(first_item)
                        movie_name = movie_names[idx]
                elif isinstance(first_item, str):
                    movie_name = first_item
                elif isinstance(first_item, dict):
                    movie_name = first_item.get("movie_name")
            if not movie_name:
                #fallback: try to extract a known movie name from the user context
                for utter in context:
                    for name in movie_names:
                        if name in utter:
                            movie_name = name
                            break
                    if movie_name:
                        break
            if not movie_name:
                movie_name = "a movie"

            #KB lookup and debug print
            knowledge = ""
            if movie_name and movie_name in movie_kb:
                kb_entry = movie_kb[movie_name]
                knowledge = build_short_kb(kb_entry, max_entities=2, max_review_sents=2)
            # Short debug print
            if knowledge:
                kb_preview = knowledge.replace("\n", " | ")[:120] + ("..." if len(knowledge) > 120 else "")
                print(f"[DEBUG] KB for {movie_name}: {kb_preview}")
            else:
                print(f"[DEBUG] No KB for {movie_name}")

            user_emotion = agent1_result.get("emotion", "")
            response = agent2.generate_response(
                dialogue_context=context,
                user_emotion=user_emotion,
                knowledge=knowledge,
                movie_name=movie_name,
                max_new_tokens=120
            )
            entry = {
                "context": context,
                "user_state": agent1_result,
                "recommended_items": rec_items,
                "knowledge_used": knowledge,
                "response": response,
                "ground_truth_response": sample.get("resp", ""),
                "ground_truth_rec": sample.get("rec", None),
                "agent_logs": {},
            }
        else:
            #DIALOGPT LOGIC
            agent2_result = agent2.process(
                context,
                agent1_result  # user_state: 'intent', 'intent_score', 'emotion', 'emotion_score'
            )
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

    #saving results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(results)} outputs to: {args.output_file}")

if __name__ == "__main__":
    main()
