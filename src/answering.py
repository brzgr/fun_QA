import logging
from typing import List


def get_question_embedding(question: str, embedder) -> List[float]:
    question_embedding = embedder.embed(question)
    return question_embedding


def get_similarity(question: List[str], memory, embedder):
    question_embedding = get_question_embedding(question=question, embedder=embedder)
    scores, indices = memory.index.search(question_embedding, 3)
    return scores, indices


def retrieve_relating_info(question: str, memory, embedder) -> str:
    scores, indices = get_similarity(
        question=[question], memory=memory, embedder=embedder
    )
    indices = list(indices[0])
    relating_info = [memory.corpus_chunks[i] for i in indices]
    return "\n".join(relating_info)
