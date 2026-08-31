"""On-disk JSONL cache for retrieval responses.

Thread-safe (file lock). Keyed by (index, query, k); value is the raw list of
hits returned by the backend.
"""
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any


class JsonCache:
    """Lazy JSON cache backed by a single .jsonl file.

    Each line is one record: {"key": "<sha>", "data": <any>}.
    On first .get() / .set() we lazy-load into memory.
    Writes append + flush; in-memory map is updated atomically under a lock.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _hash(parts: tuple) -> str:
        s = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def _ensure_loaded(self):
        if self._mem is not None:
            return
        m: dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        m[rec["key"]] = rec["data"]
                    except Exception:
                        pass
        self._mem = m

    def get(self, *parts) -> Any | None:
        with self._lock:
            self._ensure_loaded()
            assert self._mem is not None
            return self._mem.get(self._hash(parts))

    def set(self, *parts, data: Any) -> None:
        key = self._hash(parts)
        rec = {"key": key, "data": data}
        with self._lock:
            self._ensure_loaded()
            assert self._mem is not None
            self._mem[key] = data
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        with self._lock:
            self._ensure_loaded()
            assert self._mem is not None
            return len(self._mem)
