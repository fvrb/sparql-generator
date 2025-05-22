import json
import numpy as np
from openai import OpenAI
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

client = OpenAI(
  api_key="your-key"
)

def get_entities(text):
    headers = {
        "Accept": "application/json",
    }
    req_data = {
        "text": text,
        "language": {"lang": "en"}
    }
    response = requests.post(
        "http://localhost:8090/service/disambiguate",
        params={"language": "en"},
        headers=headers,
        json=req_data
    )

    data = response.json()
    entities = []
    for ent in data.get("entities", []):
        if "rawName" in ent and "wikidataId" in ent:
            entities.append({
                "text": ent["rawName"],
                "wikidataId": ent["wikidataId"]
            })
    return entities

def load_data():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    with open('lc-quad2.json', 'r') as f:
        data = json.load(f)
        data = [obj for obj in data if "question" in obj and obj["question"]]

    return embeddings, data

def get_examples(nlq, n, embeddings, data):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_embedding = model.encode([nlq])[0]

    similarity = cosine_similarity([q_embedding], embeddings)[0]
    closest_indices = np.argsort(similarity)[-n:][::-1]

    most_similar = [data[i] for i in closest_indices]
    examples = [{"question": obj["question"], "sparql": obj["sparql_wikidata"]} for obj in most_similar]

    return examples


def generate_sparql(nlq, entities, examples, cot):
    entity_str = "\n".join([f"- {e['text']} → {e['wikidataId']}" for e in entities])
    examples_str = "\n".join([f"Example #{ind}:\nQuestion: {ex['question']}\nSPARQL: {ex['sparql']}\n" for ind, ex in enumerate(examples)])


    prompt = f"""
Write a SPARQL query that answers the question using Wikidata.

Given this natural language question:
"{nlq}"

With the following entities:
{entity_str}

And these examples of similar question-query pairs:
{examples_str}

{"Let's think step by step." if cot else ""}
"""
    print(f"Generated prompt:\n{prompt}")

    response = client.chat.completions.create(
      model="gpt-4.1-2025-04-14",
      store=True,
      messages=[
        {
            "role": "system",
            "content": "Your task is to generate accurate SPARQL queries that answer natural language questions based on provided entities and example pairs. Use only Wikidata properties and structure your queries correctly. Your responses must include only the SPARQL query code, with no additional explanation or formatting."
        },
        {
            "role": "user",
            "content": prompt
        }
      ]
    )
    return response.choices[0].message.content.strip()


def run_pipeline(nlq, embeddings, data, rag_n, cot):
    print("Extracting entities...")
    entities = get_entities(nlq)
    print(f"Entities found: {entities}")

    print("Generating examples...")
    examples = get_examples(nlq, rag_n, embeddings, data)
    print(examples)

    print("Generating SPARQL query...")
    sparql_query = generate_sparql(nlq, entities, examples, cot)
    return sparql_query


if __name__ == "__main__":
    question = "What countries border Croatia?"
    embeds, lc = load_data()
    sparql = run_pipeline(question, embeds, lc, 3, True)
    print("\nGenerated SPARQL query:\n")
    print(sparql)
