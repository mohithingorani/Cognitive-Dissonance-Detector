import spacy
from typing import List

nlp = spacy.load('en_core_web_sm')


def split_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def pos_tags(text: str):
    doc = nlp(text)
    return [(t.text, t.pos_, t.dep_) for t in doc]


if __name__ == '__main__':
    print(split_sentences('I love it but I also hate it.'))
