import sys
sys.path.append('/data/s3905993/ECRHMAS/src')
from agents.llama_response_agent import LlamaResponseAgent

if __name__ == "__main__":
    model_dir = "/data/s3905993/ECRHMAS/src/models/llama2_chat"
    agent = LlamaResponseAgent(model_dir, device="cuda", debug=True)
    prompt = (
        "<s>[INST] <<SYS>>\nYou are an empathetic movie assistant. "
        "Recommend a movie for a user who is feeling nostalgic. Be concise, empathetic, and mention the movie title.\n<</SYS>>\n"
        "User: Hi! I miss movies from my childhood. Any suggestions?\nAssistant:"
    )
    response = agent.generate(prompt, max_new_tokens=80)
    print("\nGenerated response:\n", response)
