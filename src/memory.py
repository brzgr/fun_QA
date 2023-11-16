import pickle
from typing import List

import faiss


class Memory:
    def __init__(
        self, faiss_index=None, corpus_chunks: List[str] = None, model_dim: int = 384
    ):
        if not faiss_index:
            self.index = faiss.IndexFlatL2(model_dim)
            self.corpus_chunks = []
        else:
            self.index = faiss_index
            self.corpus_chunks = corpus_chunks

    def add(self, embeddings: List[float], chunk: List[str]):
        self.index.add(embeddings)
        self.corpus_chunks += chunk

    def save(self):
        faiss.write_index(self.index, "memory.index")
        with open("data.pkl", mode="wb") as file:
            chunks = pickle.dump(self.corpus_chunks, file)

    @classmethod
    def from_faiss_checkpoint(cls, faiss_checkpoint_path: str):
        index = faiss.read_index(faiss_checkpoint_path)
        with open("data.pkl", mode="rb") as file:
            chunks = pickle.load(file)
        return cls(faiss_index=index, corpus_chunks=chunks)
