from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==========================================
# 1. LOAD DOCUMENT
# ==========================================

loader = TextLoader("data/company_policy.txt")

documents = loader.load()

print(f"Documents loaded: {len(documents)}")


# ==========================================
# 2. CHUNK DOCUMENT
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=30
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# ==========================================
# 3. CREATE EMBEDDINGS
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 4. CREATE VECTOR STORE
# ==========================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created.")



# ==========================================
# 6. RETRIEVE RELEVANT CHUNKS
# ==========================================

query = "What is the sick leave policy?"

results = vectorstore.similarity_search_with_score(
    query,
    k=2
)


# ==========================================
# 7. BUILD CONTEXT
# ==========================================

context = "\n\n".join(
    document.page_content
    for document, score in results
)


# ==========================================
# 8. DISPLAY CONTEXT
# ==========================================

print("\n================================")
print("CONTEXT SENT TO LLM")
print("================================")

print(context)