import os
import gradio as gr
import requests
import json
from pathlib import Path
from typing import List, Optional
from qdrant_client import QdrantClient, models


HAYHOOKS_URL = os.getenv("HAYHOOKS_URL", "http://localhost:1416")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

INDEX_NAME = "index"
MODEL_NAME = "query"

qdrant_client = QdrantClient(url=QDRANT_URL)


def chat(message, history):
    """Handle chat messages"""
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
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    chunk = decoded_line[5:].strip()
                    if chunk != "[DONE]":
                        try:
                            data = json.loads(chunk)
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                chunks.append(delta["content"])
                                yield "".join(chunks)
                        except:
                            pass
