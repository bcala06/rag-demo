# RAG Proof-of-Concept

This project is a Retrieval-Augmented Generation (RAG) proof-of-concept that combines open-source tools for document indexing and conversational AI. It features hybrid retrieval with reranking and a user-friendly interface.

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

3. Access the services through the following URLs:

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

### Hosting

- For web hosting, expose only the Gradio app (`0.0.0.0:7860` by default).
- All model inference and retrieval happen on the host machine.
- Reasoning is enabled by default for `deepseek-r1` which may take longer for responses.

### Offline Usage

FastEmbed models, nltk, and tiktoken are cached inside the project directory. This way, the app can be configured to run indexing and querying offline (no downloads needed).

The following steps are only required for offline usage:

1. Compose/run the app with an internet connection at least once beforehand. The following automatically cached inside `hayhooks\cache\`:

   - `models\fastembed\`
   - `tiktoken`

2. Download and place `nltk_data` inside `hayhooks\cache\`. You can get the files by following the steps [here](https://www.nltk.org/data.html).

3. Set the parameters for the FastEmbed models in `hayhooks\components\pipelines.py` as follows:

   ```python
   sparse_doc_embedder = FastembedSparseDocumentEmbedder(local_files_only=True)
   sparse_query_embedder = FastembedSparseTextEmbedder(local_files_only=True)
   ranker = FastembedRanker(local_files_only=True)
   ```

4. Compose/recompose the app.
