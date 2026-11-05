# 풀이 기록과 후보 문항을 받아 각 문항의 정답 확률을 돌려주는 FastAPI 서버

import json
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import DKT

ROOT = Path(__file__).resolve().parent.parent
MAXLEN = 200

qmap = {int(k): v for k, v in json.load(open(ROOT / "models" / "qmap.json")).items()}
model = DKT(len(qmap))
model.load_state_dict(torch.load(ROOT / "models" / "dkt_fold0.pt", map_location="cpu"))
model.eval()

app = FastAPI(title="edu_tutor KT API")


class Interaction(BaseModel):
    qid: int
    response: int = Field(ge=0, le=1)


class PredictRequest(BaseModel):
    history: list[Interaction]
    candidates: list[int]


@app.get("/health")
def health():
    return {"status": "ok", "n_questions": len(qmap)}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    hist = [x for x in req.history if x.qid in qmap][-MAXLEN:]
    if not hist:
        raise HTTPException(400, "유효한 풀이 기록이 없습니다.")
    bad = [c for c in req.candidates if c not in qmap]
    if bad:
        raise HTTPException(400, f"알 수 없는 문항 ID: {bad[:5]}")

    q = torch.tensor([[qmap[x.qid] for x in hist]])
    r = torch.tensor([[x.response for x in hist]])
    with torch.no_grad():
        probs = model.predict_all(q, r)[0]

    preds = [{"qid": c, "p_correct": round(float(probs[qmap[c]]), 4)} for c in req.candidates]
    return {"predictions": preds, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}