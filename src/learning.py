import logging
import os

import tqdm

from wikipedia_utils import get_topic_data
from utils import slice_corpus_generator

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def learn_topic(topic: str, memory, embedder):
    topic_data = get_topic_data(topic=topic)
    # logging.info(topic_data)
    sliced_data = slice_corpus_generator(corpus=topic_data)
    logging.info("Memorizing new information!")
    for chunk in tqdm.tqdm(sliced_data):
        embeddings = embedder.embed(chunk)
        memory.add(embeddings, chunk)
    logging.info("New information added!!")


if __name__ == "__main__":
    learn_topic("Apex Legends")
