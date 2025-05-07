import pymongo
import requests

client = pymongo.MongoClient("mongodb+srv")
db = client.sample_mflix
collection = db.movies

# items= collection.find().limit(5)
# for item in items:
#     print(item)

hf_token = ""
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:

  response = requests.post(
    embedding_url,
    headers={"Authorization": f"Bearer {hf_token}"},
    json={"inputs": text})

  if response.status_code != 200:
    raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

  return response.json()

for doc in collection.find({'plot':{"$exists": True}}).limit(50):
  doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
  collection.replace_one({'_id': doc['_id']}, doc)
