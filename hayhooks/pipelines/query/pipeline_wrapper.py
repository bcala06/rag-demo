import logging
from typing import Generator, List, Union

from components.pipelines import create_query_pipeline
from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer

from hayhooks import (
    BasePipelineWrapper,
    get_last_user_message,
    log,
    streaming_generator,
)

# LOGGING
logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)
tracing.tracer.is_content_tracing_enabled = True
tracing.enable_tracing(LoggingTracer(tags_color_strings={"haystack.component.input": "\x1b[1;31m", "haystack.component.name": "\x1b[1;34m"}))


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = create_query_pipeline()

    def run_api(self, query: str) -> str:
        log.trace(f"Running pipeline with prompt: {query}")
        result = self.pipeline.run({
                "dense_query_embedder": {"text": query},
                "sparse_query_embedder": {"text": query},
                "ranker": {"query": query},
                "prompt_builder": {"query": query},
            })
        return result["generator"]["replies"][0]

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        query = get_last_user_message(messages)
        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "dense_query_embedder": {"text": query},
                "sparse_query_embedder": {"text": query},
                "ranker": {"query": query},
                "prompt_builder": {"query": query},
            })
