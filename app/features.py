from sentence_transformers import SentenceTransformer
from app.labeling import polarity_score, has_contrastive_joiner
import numpy as np


SENT_ENCODER = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_embedding(text: str):
    return SENT_ENCODER.encode(text)


def affective_features(text: str):
    # polarity and subjectivity approximated by TextBlob (can be replaced)
    from textblob import TextBlob
    tb = TextBlob(text)
    return {
        'polarity': tb.sentiment.polarity,
        'subjectivity': tb.sentiment.subjectivity,
        'joiner': float(has_contrastive_joiner(text))
    }


def linguistic_features(text: str):
    # simple features: length, punctuation counts, negations
    negs = sum(1 for w in text.lower().split() if w in ['not', 'no', 'never', 'none', 'nothing'])
    return {
        'len': len(text),
        'num_commas': text.count(','),
        'num_question': text.count('?'),
        'negations': negs
    }


def featurize(text: str):
    emb = semantic_embedding(text)
    aff = affective_features(text)
    ling = linguistic_features(text)
    # return concatenated vector and dict for interpretability
    vec = np.concatenate([
        emb,
        np.array([
            aff['polarity'],
            aff['subjectivity'],
            aff['joiner'],
            ling['len'],
            ling['num_commas'],
            ling['num_question'],
            ling['negations']
        ])
    ])
    meta = {**aff, **ling}
    return vec, meta