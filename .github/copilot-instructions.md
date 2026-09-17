# RAG Project Coding Instructions

## Project Goal
This project is for learning and implementing Retrieval-Augmented Generation (RAG).

## Architecture
Document
→ Chunking
→ Embeddings
→ FAISS Vector Store
→ Retriever
→ Context
→ LLM
→ Answer

## Coding Guidelines
- Use Python.
- Prefer clear, beginner-friendly code.
- Explain important RAG concepts in comments.
- Keep retrieval and generation logically separate.
- Use Hugging Face embeddings locally.
- Use FAISS for vector retrieval.
- Use Ollama with Llama 3.2 for local LLM generation.
- Do not hardcode API keys or secrets.
- Never expose `.env` contents.
- Preserve existing working functionality when modifying code.

## Learning Rule
When suggesting code, explain:
1. What the code does.
2. Why it is needed.
3. Where it fits in the RAG pipeline.

Do not generate an entire project unless explicitly requested.