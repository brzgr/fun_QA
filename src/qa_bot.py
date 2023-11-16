import logging
import os

from learning import learn_topic
from answering import retrieve_relating_info
from memory import Memory
from models import Embedder, RobertaLM

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    # init memory
    if os.path.exists("memory.index"):
        memory = Memory.from_faiss_checkpoint("memory.index")
    else:
        memory = Memory()

    # init embedder
    embedder = Embedder()

    # init LLM
    llm = RobertaLM()
    question = input("Enter your question: ")
    learn_topic(topic=question, memory=memory, embedder=embedder)
    # memory.save()
    relating_info = retrieve_relating_info(question, memory, embedder)
    answer = llm.answer(context=relating_info, question=question)
    print(answer)


if __name__ == "__main__":
    main()
