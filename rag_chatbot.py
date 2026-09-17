from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


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
# 5. CREATE RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# ==========================================
# 6. CREATE LOCAL LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ==========================================
# 7. INTERACTIVE RAG CHATBOT
# ==========================================

while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    # --------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------

    results = retriever.invoke(question)

    # --------------------------------------
    # Display retrieved chunks
    # --------------------------------------

    print("\n================================")
    print("RETRIEVED CONTEXT")
    print("================================")

    for i, document in enumerate(results, start=1):
        print(f"\n--- Chunk {i} ---")
        print(document.page_content)

    # --------------------------------------
    # Build context
    # --------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # --------------------------------------
    # Create prompt
    # --------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is present in the context, answer directly.
4. If the answer is not present, say:
   "I don't know based on the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # --------------------------------------
    # Generate answer
    # --------------------------------------

    response = llm.invoke(prompt)

    # --------------------------------------
    # Display answer
    # --------------------------------------

    print("\n================================")
    print("FINAL ANSWER")
    print("================================")

    print(response.content)