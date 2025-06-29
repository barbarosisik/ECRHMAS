# MA-ERGen: Multi-Agent Empathetic Response Generation for Conversational Systems

This project is my MSc thesis implementation at Leiden University, exploring if a modular multi-agent architecture—**intent & emotion recognition → knowledge-aware empathetic response generation**—can outperform standard single-agent methods in empathy and recommendation quality.

## Structure

- **data/**: Redial, EmpatheticDialogues, movie KB, and processed files  
- **src/agents/**: Modular agent definitions (intent/emotion recognizer, Llama/DialoGPT responder, RL critic)
- **scripts/**: Preprocessing, training, evaluation scripts
- **notebooks/**: Exploratory Jupyter notebooks
- **experiments/** and **results/**: Logs, evaluation outputs, and metrics

## Datasets Used

- **Redial** (Movie recommendation dialogues)
- **EmpatheticDialogues** (Emotion-labeled conversations)
- Movie knowledge base (external data, entities, reviews)

---

All code and documentation are provided for reproducibility.  
For issues, contact: barbarosisik [at] umail.leidenuniv.nl

