from sentence_transformers import SentenceTransformer
from endee import EndeeClient

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Endee
client = EndeeClient()

# Optional: reset DB to avoid duplicates
client.reset()

# Load data
with open("data.txt", "r") as file:
    documents = file.read().split("\n")

# Store in Endee
for doc in documents:
    emb = model.encode(doc)

    client.add({
        "vector": emb.tolist(),
        "text": doc
    })

print("✅ Data stored in Endee")

# Search function
def search(query):
    query_embedding = model.encode(query)

    results = client.search({
        "vector": query_embedding.tolist(),
        "top_k": 1
    })

    return results[0]["text"]

# User input
query = input("Enter your question: ")
result = search(query)

print("\nBest Match:", result)