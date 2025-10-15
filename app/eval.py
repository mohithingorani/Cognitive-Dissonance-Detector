import numpy as np
import pandas as pd
import torch
from app.features import featurize
from app.labeling import weak_label_dissonance


def predict_texts(texts):
    X = []
    metas = []
    for t in texts:
        v, m = featurize(t)
        X.append(v)
        metas.append(m)
    X = np.vstack(X)
    Xt = torch.tensor(X, dtype=torch.float32)
    mdl = torch.load('data/dissonance_model.pt', map_location='cpu')
    from app.model import MLP
    model = MLP(in_dim=X.shape[1])
    model.load_state_dict(mdl['state'])
    model.eval()
    with torch.no_grad():
        pred = model(Xt).detach().cpu().numpy().reshape(-1)
    return pred, metas


if __name__ == '__main__':
    df = pd.read_csv('data/sample_texts.csv')
    preds, metas = predict_texts(df['text'].tolist())
    heur = [weak_label_dissonance(t) for t in df['text']]
    for t, p, h in zip(df['text'], preds, heur):
        print(f'TEXT: {t}\nPRED={p:.3f} HEUR={h:.3f}\n---')
