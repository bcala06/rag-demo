# RAG Proof-of-Concept

This project is a hybrid Retrieval-Augmented Generation (RAG) proof-of-concept that combines open-source tools for document indexing and conversational AI. It features hybrid retrieval with reranking and a user-friendly interface.

## Overview

All of the backend components are managed with Haystack. Hayhooks is used to manage the API for the pipeline. Gradio is used as the interface to interact with the Hayhooks and Qdrant endpoints.

| Component                | Model                                     | Source    |
| ------------------------ | ----------------------------------------- | --------- |
| Vector Storage/Retriever | Qdrant Vector Database                    | Qdrant    |
| Document Converter       | Tika                                      | Apache    |
| Dense Embedder           | `nomic-embed-text:v1.5`                   | Ollama    |
| Sparse Embedder          | `Qdrant/bm42-all-minilm-l6-v2-attentions` | FastEmbed |
| Reranker                 | `jinaai/jina-reranker-v1-turbo-en`        | FastEmbed |
| Generator                | `deepseek-r1:latest`                      | Ollama    |

## Getting Started

### Requirements

- Docker
- Docker Compose
- Ollama

### Running the App

1. Run Ollama on http://localhost:11434

   - `ollama pull nomic-embed-text:v1.5`
   - `ollama pull deepseek-r1:latest`
   - `ollama serve`

2. Inside the project root directory, compose with docker:

   - `docker compose up`

3. Access the services via your browser:

   | Service  | URL                        |
   | -------- | -------------------------- |
   | Gradio   | http://localhost:7860      |
   | Hayhooks | http://localhost:1416/docs |
   | Qdrant   | http://localhost:8333      |

## Gradio Interface

The interface contains two tabs:

### 1. Indexing

- Upload documents for embedding and storage in Qdrant.
- Remove previously indexed files from the database.

### 2. Chat

- Ask questions about the uploaded documents.
- The RAG LLM responds based on retrieved context from indexed files.

## Notes

- To host the app online, expose the Gradio app from `0.0.0.0:7860`.
- All model inference and retrieval happen on the host machine.
- Reasoning is enabled for `deepseek-r1` which may take longer for responses.
