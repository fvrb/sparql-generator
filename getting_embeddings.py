import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

with open("lc-quad2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data if "question" in item and item["question"]]

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(questions, show_progress_bar=True)

embeddings_array = np.array(embeddings)

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_array, f)

print(f"Processed {len(questions)} questions and saved embeddings.")
