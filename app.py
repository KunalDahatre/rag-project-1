from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

print("API key loaded:", bool(key))
print("Using placeholder:", key == "your_api_key_here")


# ==========================================
# 1. LOAD DOCUMENT
# ==========================================

loader = TextLoader("data/company_policy.txt")
documents = loader.load()

print(f"Documents loaded: {len(documents)}")


# ==========================================
# 2. SPLIT DOCUMENT INTO CHUNKS
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

print("Embedding model loaded.")


# ==========================================
# 4. CREATE FAISS VECTOR STORE
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
# 6. LOAD LOCAL LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("Llama 3.2 loaded.")


# ==========================================
# 7. INTERACTIVE RAG CHATBOT
# ==========================================

print("\n================================")
print("RAG CHATBOT READY")
print("================================")
print("Ask questions about the company policy.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    # Exit condition
    if question.lower() in ["exit", "quit", "bye"]:
        print("\nGoodbye!")
        break

    # ======================================
    # RETRIEVE RELEVANT DOCUMENTS
    # ======================================

    results = retriever.invoke(question)

    # ======================================
    # CREATE CONTEXT
    # ======================================

    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # ======================================
    # CREATE PROMPT
    # ======================================

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
--------------------
{context}
--------------------

USER QUESTION:
{question}

ANSWER:
"""

    # ======================================
    # GENERATE ANSWER
    # ======================================

    response = llm.invoke(prompt)

    print("\nAI:", response.content)
    print()