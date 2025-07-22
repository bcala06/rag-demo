import os
import logging
from pathlib import Path
from typing import Generator, List, Union

from haystack import Document, Pipeline
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.converters import TikaDocumentConverter
from haystack.components.preprocessors import RecursiveDocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder, ChatPromptBuilder
from haystack.components.rankers import MetaFieldRanker

# Qdrant components
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantSparseEmbeddingRetriever,
    QdrantHybridRetriever 
)

# FastEmbed components
from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack_integrations.components.embedders.fastembed import (
    FastembedTextEmbedder,
    FastembedDocumentEmbedder,
    FastembedSparseTextEmbedder,
    FastembedSparseDocumentEmbedder
)

# Ollama components
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder, OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator, OllamaChatGenerator


###################################################################################################


ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
tika_url = os.getenv("TIKA_URL", "http://localhost:9998/tika")

embedding_name = "index"
embedding_dim = 768

dense_embedder_model = "nomic-embed-text:v1.5"
sparse_embedder_model = "Qdrant/bm42-all-minilm-l6-v2-attentions"

ranker_model = "jinaai/jina-reranker-v1-turbo-en"
generator_model = "deepseek-r1:latest"


###################################################################################################


def create_document_store() -> QdrantDocumentStore:
    return QdrantDocumentStore(
        url=qdrant_url,
        index=embedding_name,
        embedding_dim=embedding_dim,
        recreate_index=False, # enable only to recreate the index and not connect to the existing one
        use_sparse_embeddings=True,
        sparse_idf=True,
        on_disk=True,
    )


def create_index_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    converter = TikaDocumentConverter(tika_url=tika_url)
    cleaner = DocumentCleaner()

    chunker = RecursiveDocumentSplitter(
        split_length=1000,
        split_overlap=0,
        split_unit="token",
        separators=["\n\n", "sentence", "\n", " "],
        sentence_splitter_params={
            "language": "en",
            "use_split_rules": True,
            "keep_white_spaces": True,
        },
    )
    chunker.warm_up()

    dense_doc_embedder = OllamaDocumentEmbedder(
        model=dense_embedder_model,
        url=ollama_url,
    )
    
    sparse_doc_embedder = FastembedSparseDocumentEmbedder(
        model=sparse_embedder_model,
        local_files_only=True, # enable to only use cached models
    )
    sparse_doc_embedder.warm_up()

    writer = DocumentWriter(
        document_store=document_store, 
        policy=DuplicatePolicy.OVERWRITE,
    )

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", converter)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("chunker", chunker)
    indexing_pipeline.add_component("dense_embedder", dense_doc_embedder)
    indexing_pipeline.add_component("sparse_embedder", sparse_doc_embedder)
    indexing_pipeline.add_component("writer", writer)

    indexing_pipeline.connect("converter.documents", "cleaner.documents")
    indexing_pipeline.connect("cleaner.documents", "chunker.documents")
    indexing_pipeline.connect("chunker.documents", "dense_embedder.documents")
    indexing_pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    indexing_pipeline.connect("sparse_embedder.documents", "writer.documents")

    return indexing_pipeline


def create_query_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    try:
        script_dir = Path(__file__).parent
        prompt_path = script_dir / "rag_prompt.txt"
        prompt_template = [ChatMessage.from_user(prompt_path.read_text(encoding="utf-8"))]
    except FileNotFoundError:
        raise RuntimeError("Prompt template file 'rag_prompt.txt' not found")

    prompt_builder = ChatPromptBuilder(template=prompt_template, required_variables=["context", "query"])

    dense_query_embedder = OllamaTextEmbedder(
        model=dense_embedder_model,
        url=ollama_url,
    )

    sparse_query_embedder = FastembedSparseTextEmbedder(
        model=sparse_embedder_model,
        local_files_only=True, # enable to only use cached models
    )
    sparse_query_embedder.warm_up()
    
    retriever = QdrantHybridRetriever(
        document_store=document_store,
        top_k=10,
        score_threshold=0.5,
    )

    ranker = FastembedRanker(
        model_name=ranker_model,
        top_k=5,
        local_files_only=True, # only use models from "/cache/models/fastembed"
    )
    ranker.warm_up()

    # reranks based on filename relevance to query
    meta_ranker = MetaFieldRanker(
        meta_field="file_path",
        weight=0.5,
        top_k=3,
    )

    generator = OllamaChatGenerator(
        model=generator_model,
        url=ollama_url,
        generation_kwargs={
            "num_predict": -1,
            "temperature": 0.5,
            "n_ctx": 8192,
        }
    )

    query_pipeline = Pipeline()
    query_pipeline.add_component("dense_query_embedder", dense_query_embedder)
    query_pipeline.add_component("sparse_query_embedder", sparse_query_embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.add_component("ranker", ranker)
    query_pipeline.add_component("meta_ranker", meta_ranker)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("generator", generator)

    query_pipeline.connect("dense_query_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("sparse_query_embedder.sparse_embedding", "retriever.query_sparse_embedding")
    query_pipeline.connect("retriever.documents", "ranker.documents")
    query_pipeline.connect("ranker.documents", "meta_ranker.documents")
    query_pipeline.connect("meta_ranker.documents", "prompt_builder.context")
    query_pipeline.connect("prompt_builder.prompt", "generator.messages")

    return query_pipeline
