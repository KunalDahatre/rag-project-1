from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create embeddings
texts = [
    "What is RAG?",
    "Explain retrieval augmented generation.",
    "How does a vector database work?",
    "What is the weather today?"
]

vectors = embeddings.embed_documents(texts)

# Convert to numpy arrays
vectors = np.array(vectors)

# Query
query = "How many annual leave days do employees get?"

query_vector = np.array(
    embeddings.embed_query(query)
)

# Calculate cosine similarity
similarities = np.dot(vectors, query_vector) / (
    np.linalg.norm(vectors, axis=1) *
    np.linalg.norm(query_vector)
)

# Display results
for text, score in zip(texts, similarities):
    print(f"{score:.4f} → {text}")