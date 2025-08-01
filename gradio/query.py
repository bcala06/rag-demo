import json
import os

import requests
from qdrant_client import QdrantClient

import gradio as gr

HAYHOOKS_URL = os.getenv("HAYHOOKS_URL", "http://localhost:1416")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

INDEX_NAME = "index"
MODEL_NAME = "query"

qdrant_client = QdrantClient(url=QDRANT_URL)


def chat(message, history):
    """Handle chat messages"""
    
    if not len(message):
        raise gr.Error("Chat messages cannot be empty")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }
    
    with requests.post(
        f"{HAYHOOKS_URL}/{MODEL_NAME}/chat",
        json=payload,
        stream=True
    ) as response:
        response.raise_for_status()
        
        chunks = []
        message = ""
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    chunk = decoded_line[5:].strip()
                    try:
                        data = json.loads(chunk)
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            if delta["content"] != "":
                                chunks.append(delta["content"])
                                message = "".join(chunks)
                                yield message
                    except Exception:
                        pass