import argparse


def load_and_featurize(path='data/sample_texts.csv'):
    df = pd.read_csv(path)
    X = []
    metas = []
    for t in df['text'].tolist():
        v, m = featurize(t)
        X.append(v)
        metas.append(m)
    X = np.vstack(X)
    return X, df


def train_demo():
    X, df = load_and_featurize()
    # weak labels
    from app.labeling import weak_label_dissonance
    y = np.array([weak_label_dissonance(t) for t in df['text']])

    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.4, random_state=42)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xv = torch.tensor(Xv, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    yv = torch.tensor(yv, dtype=torch.float32).unsqueeze(1)

    model = MLP(in_dim=Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for ep in range(150):
        model.train()
        opt.zero_grad()
        out = model(Xtr)
        loss = loss_fn(out, ytr)
        loss.backward()
        opt.step()
        if ep % 30 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(Xv)
                val_loss = loss_fn(pred, yv).item()
            print(f'ep {ep} train_loss={loss.item():.4f} val_loss={val_loss:.4f}')

    torch.save({'state': model.state_dict()}, 'data/dissonance_model.pt')
    print('Saved demo model to data/dissonance_model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    if args.demo:
        train_demo()
    else:
        print('Run with --demo to train on sample data')
