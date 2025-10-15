from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
from app.eval import predict_texts


app = FastAPI()


class PredictIn(BaseModel):
	text: str


class ExplainOut(BaseModel):
	dissonance: float
	heuristic: float
	meta: dict


@app.post('/predict', response_model=ExplainOut)
def predict(inp: PredictIn):
	preds, metas = predict_texts([inp.text])
	from app.labeling import weak_label_dissonance
	h = weak_label_dissonance(inp.text)
	return {'dissonance': float(preds[0]), 'heuristic': float(h), 'meta': metas[0]}


@app.get('/health')
def health():
	return {'status':'ok'}


if __name__=='__main__':
	uvicorn.run(app, host='0.0.0.0', port=8000)