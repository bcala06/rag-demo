import os
from pathlib import Path
from typing import List

import requests
from qdrant_client import QdrantClient, models

HAYHOOKS_URL = os.getenv("HAYHOOKS_URL", "http://localhost:1416")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

INDEX_NAME = "index"
MODEL_NAME = "query"

qdrant_client = QdrantClient(url=QDRANT_URL)


def upload_files(files: List[str]) -> str:
    """Upload files to Hayhooks server"""
    file_objs = []  
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        file_objs.append(("files", (path.name, open(file_path, "rb"))))
    
    response = requests.post(
        f"{HAYHOOKS_URL}/index/run",
        files=file_objs
    )
    
    for _, (_, file_obj) in file_objs:
        file_obj.close()
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")
    
    return response.json()


def get_uploaded_files(limit: int = 100) -> List[str]:
    """Retrieve list of already uploaded files from Qdrant"""
    try:
        records = qdrant_client.scroll(
            collection_name=INDEX_NAME,
            limit=limit,
            with_payload=True
        )
        
        uploaded_files = set()
        for record in records[0]:
            payload = record.payload
            if payload and 'meta' in payload and 'file_path' in payload['meta']:
                uploaded_files.add(payload['meta']['file_path'])
        
        return sorted(list(uploaded_files))
        
    except Exception as e:
        print(f"Error fetching uploaded files: {e}")
        return []


def delete_files(file_names: List[str]) -> tuple[str, list]:
    """Delete files from Qdrant collection by their names"""
    if not file_names:
        return "No files selected for deletion.", []
    try:
        deleted_count = 0
        for file_name in file_names:
            qdrant_client.delete(
                collection_name=INDEX_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                                key="meta.file_path",
                                match=models.MatchValue(value=file_name),
                            ),
                        ],
                    )
                ),
            )
            deleted_count += 1
        return f"Successfully deleted {deleted_count} file(s): {', '.join(file_names)}", []
    except Exception as e:
        return f"Deletion failed: {str(e)}", file_names


def process_upload(file_objs) -> tuple[str, list]:
    """Handle file upload and return results"""
    if not file_objs:
        return "No files selected for upload.", []
    try:
        file_paths = [file_obj.name for file_obj in file_objs]
        result = upload_files(file_paths)
        output_msg = f"Files uploaded: {', '.join(file_paths)}\n\nIndexing API Response: {result}\n\n"
        return output_msg, []
    except Exception as e:
        return f"Upload failed: {str(e)}", []
