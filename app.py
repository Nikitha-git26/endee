from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
with open("data.txt", "r") as file:
    documents = file.read().split("\n")

# Convert documents to embeddings
doc_embeddings = model.encode(documents)

# ---- Simulated Endee DB (vector storage) ----
database = list(zip(documents, doc_embeddings))

# Search function
def search(query):
    query_embedding = model.encode([query])[0]
    
    similarities = []
    
    for doc, emb in database:
        sim = np.dot(emb, query_embedding)
        similarities.append(sim)
    
    best_index = np.argmax(similarities)
    return documents[best_index]

# User input
query = input("Enter your question: ")
result = search(query)

print("\nBest Match:", result)