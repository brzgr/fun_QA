from typing import List


def slice_corpus_generator(corpus: str, chunk_size: int = 256, num_chunks: int = 10):
    split_corpus = corpus.split()
    chunks = []
    counter = 0
    for i in range(0, len(split_corpus), chunk_size):
        chunk = " ".join(split_corpus[i : i + chunk_size])
        chunks.append(chunk)
        counter += 1
        if counter == num_chunks:
            yield chunks
            counter = 0
            chunks = []
