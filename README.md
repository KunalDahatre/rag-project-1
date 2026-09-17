# 🚀 RAG Project 1 — Getting Started with RAG Pipeline

A beginner-friendly implementation of a **Retrieval-Augmented Generation (RAG)** pipeline for document-based Question Answering.

This project demonstrates how an AI system can retrieve relevant information from a private document and use an LLM to generate a grounded answer.

---

## 📌 Project Overview

Traditional Large Language Models (LLMs) answer questions using the knowledge learned during training.

However, they may not know information from private or newly created documents.

**Retrieval-Augmented Generation (RAG)** solves this problem by retrieving relevant information from external documents and providing it to the LLM as context.

In this project, we build a RAG system that answers questions from a company employee policy document.

---

## 🧠 What is RAG?

RAG stands for:

> **Retrieval-Augmented Generation**

The basic process is:

```text
User Question
      ↓
Query Embedding
      ↓
Vector Similarity Search
      ↓
Retrieve Relevant Chunks
      ↓
Build Context
      ↓
Context + Question
      ↓
LLM
      ↓
Generated Answer