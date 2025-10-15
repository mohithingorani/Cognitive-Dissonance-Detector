from textblob import TextBlob
import numpy as np


# emotion polarity (valence) via TextBlob sentiment polarity
# stance heuristics: presence of contradiction words and polarity mismatch

CONTRASTING_JOINERS = ['but', 'although', 'yet', 'however', 'though', 'despite']


def polarity_score(text: str) -> float:
    return TextBlob(text).sentiment.polarity


def has_contrastive_joiner(text: str) -> bool:
    t = text.lower()
    return any(j in t for j in CONTRASTING_JOINERS)


def weak_label_dissonance(text: str) -> float:
    """Heuristic: high dissonance if contrasting joiner present and sentence-level polarity mismatch.
    Returns 0..1 score."""
    sents = [s.strip() for s in text.split(',') if s.strip()]
    if len(sents) < 2:
        # fallback to polarity magnitude
        return min(1.0, abs(polarity_score(text))) * 0.2
    # compute polarity per clause
    pols = [polarity_score(s) for s in sents]
    # dissonance: high variance + presence of joiner
    var = float(np.var(pols))
    join = 1.0 if has_contrastive_joiner(text) else 0.0
    score = min(1.0, var * 2.0 + join * 0.5)
    return score


if __name__ == '__main__':
    texts = [
        'I love my job, but I wake up dreading it',
        'I love it and it makes me happy'
    ]
    for t in texts:
        print(t, polarity_score(t), weak_label_dissonance(t))
