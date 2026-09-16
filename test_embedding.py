from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query("What is RAG?")

print("Embedding created successfully!")
print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])