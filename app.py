from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents
with open("data.txt", "r") as file:
    documents = [line.strip() for line in file if line.strip()]

# Create embeddings
doc_embeddings = model.encode(documents)

# Search function
def search(query):
    query_embedding = model.encode([query])[0]

    similarities = []
    for emb in doc_embeddings:
        sim = np.dot(emb, query_embedding)
        similarities.append(sim)

    # Get top 3 matches
    top_indices = np.argsort(similarities)[-3:][::-1]
    return top_indices

# Generate response
def generate_answer(query, indices):
    print("\n💡 Answer:")
    for i in indices:
        print("-", documents[i])

    print("\n🧠 Explanation:")
    print("This answer is generated using semantic search by comparing vector embeddings of the query and stored data.")

# Run loop
while True:
    query = input("\nAsk your question (type 'exit' to stop): ")

    if query.lower() == "exit":
        break

    results = search(query)
    generate_answer(query, results)