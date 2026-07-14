import json
import os
from typing import Any

def persist(data: dict[str, Any]):
    os.makedirs('data', exist_ok=True)
    with open('data/data.jsonl', mode='a', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, default=str)+"\n")

def persist_exception(data: dict[str, Any]):
    os.makedirs('data', exist_ok=True)
    with open('data/exceptions.jsonl', mode='a', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, default=str)+"\n")
