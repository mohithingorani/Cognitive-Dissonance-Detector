import pandas as pd
from pathlib import Path


def load_sample(path='data/sample_texts.csv'):
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(path)
	return pd.read_csv(p)


if __name__=='__main__':
	df = load_sample()
	print(df.head())