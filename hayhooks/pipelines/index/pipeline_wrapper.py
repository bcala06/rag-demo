import os
import logging
import shutil
from typing import Generator, List, Union, Optional

from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from hayhooks import BasePipelineWrapper, log
from fastapi import UploadFile, File

from components.pipelines import create_document_store, create_index_pipeline

# LOGGING
logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)
tracing.tracer.is_content_tracing_enabled = False
tracing.enable_tracing(LoggingTracer(tags_color_strings={"haystack.component.input": "\x1b[1;31m", "haystack.component.name": "\x1b[1;34m"}))


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.document_store = create_document_store()
        self.pipeline = create_index_pipeline(self.document_store)

    def run_api(self, files: Optional[List[UploadFile]] = None) -> str:
        if not files:
            return "No files provided for indexing"
        
        log.trace(f"Running pipeline with files: {[file.filename for file in files]}")
        os.makedirs("documents", exist_ok=True)
        saved_file_paths = []

        for file in files:
            file_path = os.path.join("documents", file.filename)
            file.file.seek(0)
            file_content = file.file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
            saved_file_paths.append(file_path)
        
        for file_path in saved_file_paths:
            self.pipeline.run({"converter": {"sources": [file_path]}})
        
        return "Files indexed successfully"
