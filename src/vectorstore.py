from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .pdf_ingest import Chunk

class VectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.meta: List[Chunk] = []

    def build(self, chunks: List[Chunk]):
        texts = [c.text for c in chunks]
        embs = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs.astype(np.float32))
        self.meta = chunks

    def search(self, query: str, k: int = 4) -> List[Chunk]:
        q = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q.astype(np.float32), k)
        hits = []
        for idx in I[0]:
            if idx == -1:
                continue
            hits.append(self.meta[idx])
        return hits
