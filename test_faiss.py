from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Our documents
texts = [
    "RAG stands for Retrieval Augmented Generation.",
    "A vector database stores numerical embeddings.",
    "Employees receive 18 days of annual leave.",
    "Employees can work from home with manager approval.",
    "The office working hours are 9 AM to 6 PM."
]

# 3. Create FAISS vector store
vectorstore = FAISS.from_texts(
    texts,
    embeddings
)

# 4. User question
query = "What is Retrieval Augmented Generation?"

# 5. Search for relevant documents
results = vectorstore.similarity_search_with_score(
    query,
    k=3
)

# 6. Display results
for document, score in results:
    print("\nText:", document.page_content)
    print("Distance:", score)