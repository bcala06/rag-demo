# RAG Proof-of-Concept

This project is a Retrieval-Augmented Generation (RAG) proof-of-concept that combines open-source tools for document indexing and conversational AI. It features hybrid retrieval with reranking and a user-friendly interface.

## Overview

- [Haystack](https://haystack.deepset.ai/) is used for model integration and pipeline creation.
- [Hayhooks](https://github.com/deepset-ai/hayhooks) is used to manage the API for the pipeline.
- [Gradio](https://www.gradio.app/) is used for the interface for the Hayhooks and Qdrant endpoints.

| Component                | Model                              | Source                                                 |
| ------------------------ | ---------------------------------- | ------------------------------------------------------ |
| Vector Storage/Retriever | Qdrant Vector Database             | [Qdrant](https://qdrant.tech/qdrant-vector-database/)  |
| Document Converter       | Tika                               | [Apache](https://tika.apache.org/)                     |
| Dense Embedder           | `granite-embedding:30m`            | [Ollama](https://ollama.com/library/granite-embedding) |
| Sparse Embedder          | `Qdrant/bm25`                      | [FastEmbed](https://github.com/qdrant/fastembed)       |
| Reranker                 | `jinaai/jina-reranker-v1-turbo-en` | [FastEmbed](https://github.com/qdrant/fastembed)       |
| Generator                | `gemma3n:e2b`                      | [Ollama](https://ollama.com/library/gemma3n)           |

## Getting Started

### Prerequisites

- [Docker Compose](https://docs.docker.com/compose/install/)
- [Ollama](https://ollama.com/download)

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

- Upload documents into the indexing pipeline. Uploaded documents are copied into `hayhooks\documents\`.
- Remove previously indexed files from the database.

### 2. Chat

- Ask a chatbot about the uploaded documents.
- The chatbot responds based on retrieved context from indexed files.

## Notes

### Hosting

- For web hosting, expose only the Gradio app (`0.0.0.0:7860` by default).
- All model inference and retrieval happen on the host machine.
- Hayhooks implementation is synchronous, but async is possible with some modifications.

### Performance

- The pipeline is configured to always perform retrieval based on the user's latest query. This is a limitation with the implementation that can be addressed in the future.
- Reasoning is enabled by default for `deepseek-r1` which may take longer for response generation.

### Limitations

- The pipeline is configured to retrieve documents after **every** query. This means that succeeding messages may not always be relevant to the initial query. For best results, the user should contain their entire query on a single message.
- Currently, the Gradio app only provides basic user authentication. However, more robust implementations such as [OAuth](https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers) are possible and recommended for production.

### Offline Usage

FastEmbed models, nltk, and tiktoken are cached inside the project directory. This way, the app can be configured to run indexing and querying offline (no downloads needed).

The following steps are only required for offline usage:

1. Compose/run the app with an internet connection at least once beforehand. The following should be cached in `hayhooks\cache\`: `models\fastembed\`, `tiktoken`.

2. Download and place `nltk_data` inside `hayhooks\cache\`. You can get the files by following the steps [here](https://www.nltk.org/data.html).

3. Add the parameters for the following components in `hayhooks\components\pipelines.py`:

   ```python
   sparse_doc_embedder = FastembedSparseDocumentEmbedder(local_files_only=True)
   sparse_query_embedder = FastembedSparseTextEmbedder(local_files_only=True)
   ranker = FastembedRanker(local_files_only=True)
   ```

4. Compose/recompose the app.
