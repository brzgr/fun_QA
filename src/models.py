import logging
from typing import List

from sentence_transformers import SentenceTransformer
import gpt4all
import transformers

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# all-MiniLM-L6-v2 is the fastest mode in the
# size bracket. You can also use other models
# https://www.sbert.net/docs/pretrained_models.html#model-overview


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    @property
    def model_dim(self):
        return 384

    def embed(self, corpus: List[str]) -> List[float]:
        embeddings = self.model.encode(corpus)
        assert embeddings.shape[0] == len(corpus)
        return embeddings


class GPT4AllLLM:
    def __init__(self):
        # You can also use other models
        self.llm = gpt4all.GPT4All("gpt4all-falcon-q4_0")

    def format_llm_query(self, question: str, context: str):
        # Nearest neighbour from corpus as per FAISS
        messages = f"context: {context}\n" f"question: {question}\n" f"answer: "

        return messages

    def answer(self, question: str, context: str) -> str:
        llm_in = self.format_llm_query(question=question, context=context)
        response = self.llm.generate(llm_in)
        return response


class T5LLM:
    def __init__(self):
        # You can also use other models
        self.llm = transformers.pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=64,
        )

    def format_llm_query(self, question: str, context: str):
        # Nearest neighbour from corpus as per FAISS
        messages = (
            f"{context}\n"
            f"Please answer to the following question with detail. {question}"
        )
        logging.info(messages)
        return messages

    def answer(self, question: str, context: str) -> str:
        llm_in = self.format_llm_query(question=question, context=context)
        response = self.llm(llm_in)
        return response


class RobertaLM:
    def __init__(self):
        # You can also use other models
        self.llm = transformers.pipeline("question-answering", model="deepset/roberta-base-squad2")

    def answer(self, question: str, context: str) -> str:
        response = self.llm(context=context, question=question)
        return response["answer"]
