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
# 7. CHAT BOT
# ==========================================

while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    # Retrieve relevant chunks
    results = retriever.invoke(question)

    # Build context
    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # Create prompt
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

    # Generate answer
    response = llm.invoke(prompt)

    print("\nAI:", response.content)

# ==========================================
# 8. RETRIEVE DOCUMENTS
# ==========================================

results = retriever.invoke(question)


# ==========================================
# 9. BUILD CONTEXT
# ==========================================

context = "\n\n".join(
    document.page_content
    for document in results
)

print("\n================================")
print("RETRIEVED CONTEXT")
print("================================")

print(context)

# ==========================================
# 10. CREATE PROMPT
# ==========================================

prompt = f"""
You are a document question-answering assistant.

Your task is to answer the user's question using the
information in the CONTEXT below.

IMPORTANT RULES:
1. Use the CONTEXT to answer the question.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer is clearly present in the CONTEXT,
   provide the answer directly.
5. Only say "I don't know based on the provided document"
   if the answer is genuinely not present.

CONTEXT:
--------------------
{context}
--------------------

USER QUESTION:
{question}

ANSWER:
"""


# ==========================================
# 11. SEND TO LLM
# ==========================================

response = llm.invoke(prompt)


# ==========================================
# 12. DISPLAY ANSWER
# ==========================================

print("\n================================")
print("FINAL ANSWER")
print("================================")

print(response.content)

print(context)
print(prompt)