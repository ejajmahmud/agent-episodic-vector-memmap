"""Core vector memory map implementation."""
import math
import struct
import os
from typing import List, Tuple, Dict, Any

class VectorEntry:
    def __init__(self, doc_id: str, vector: List[float], metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.vector = vector
        self.metadata = metadata

class CosineRetriever:
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if len(v1) != len(v2) or not v1:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

class VectorMemMapIndex:
    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.entries: List[VectorEntry] = []

    def add(self, doc_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dim {len(vector)} does not match index dim {self.dimension}")
        self.entries.append(VectorEntry(doc_id, vector, metadata or {}))

    def search(self, query_vec: List[float], top_k: int = 3) -> List[Tuple[str, float, Dict[str, Any]]]:
        if len(query_vec) != self.dimension:
            raise ValueError("Query dimension mismatch")

        scored = []
        for entry in self.entries:
            sim = CosineRetriever.cosine_similarity(query_vec, entry.vector)
            scored.append((entry.doc_id, sim, entry.metadata))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def export_binary(self) -> bytes:
        # Simple binary representation: [count (4B)][dim (4B)] + entries
        payload = bytearray()
        payload.extend(struct.pack("<II", len(self.entries), self.dimension))
        for e in self.entries:
            id_bytes = e.doc_id.encode("utf-8")
            payload.extend(struct.pack("<I", len(id_bytes)))
            payload.extend(id_bytes)
            for val in e.vector:
                payload.extend(struct.pack("<f", float(val)))
        return bytes(payload)
