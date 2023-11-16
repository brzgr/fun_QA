import logging
from typing import List

import wikipedia

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def search_topic(topic: str, results: int = 5) -> List[str]:
    subjects = wikipedia.search(topic)
    logging.info(subjects)
    return subjects[:5]


def get_subject_data(subject: str) -> str:
    try:
        page = wikipedia.page(title=subject, auto_suggest=False)
    except Exception as e:
        logging.warning(f"Error while trying to retrieve page {subject}. Error: {e}")
        return ""
    return page.content


def get_topic_data(topic: str) -> str:
    topic_data = []
    subjects = search_topic(topic)
    for subject in subjects:
        data = get_subject_data(subject)
        topic_data.append(data)
    return "\n".join(topic_data)


if __name__ == "__main__":
    subjects = search_topic("Apex Legends")
    logger.info(subjects)
    for subject in subjects:
        logger.info(f"Subject {subject}")
        data = get_subject_data(subject)
        logger.info(f"Data:\n{data}")
        logger.info("_" * 20)
