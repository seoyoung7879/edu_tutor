# 풀이 기록과 후보 문항을 받아 각 문항의 정답 확률을 돌려주는 FastAPI 서버

import json
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import DKT
from src import rag
from src import llm
from src import agent
import random

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
@app.get("/explain/{qid}")
def explain(qid: int):
    r = rag.get_explanation(qid)
    if r is None:
        raise HTTPException(404, "없는 문항입니다.")
    return r


@app.get("/similar/{qid}")
def similar(qid: int, k: int = 3):
    r = rag.similar(qid, k)
    if r is None:
        raise HTTPException(404, "검색 대상이 아닌 문항입니다(이미지 문항 등).")
    return {"qid": qid, "similar": r}

# 틀린 문제와 비슷한 후보를 모아 DKT 정답 확률로 골라 주는 추천 API
class RecommendRequest(BaseModel):
    history: list[Interaction]
    k: int = 3


@app.post("/recommend")
def recommend(req: RecommendRequest):
    wrong = [x.qid for x in req.history if x.response == 0][-5:]
    if not wrong:
        raise HTTPException(400, "틀린 문제가 없어 추천 기준이 없습니다.")
    done = {x.qid for x in req.history}
    cands = {}
    for w in wrong:
        for s in rag.similar(w, 10) or []:
            c = int(s["qid"])
            if c in qmap and c not in done:
                cands[c] = s["content"]
    if not cands:
        raise HTTPException(404, "추천 후보가 없습니다.")
    res = predict(PredictRequest(history=req.history, candidates=list(cands)))
    ranked = sorted(res["predictions"], key=lambda x: abs(x["p_correct"] - 0.6))
    return {"recommended": [{**r, "content": cands[r["qid"]]} for r in ranked[: req.k]]}
#문항 번호(와 선택적 질문)를 받아 LLM의 한국어 설명을 돌려주는 API
class ExplainRequest(BaseModel):
    qid: int
    question: str | None = None


@app.post("/tutor/explain")
def tutor_explain(req: ExplainRequest):
    try:
        text = llm.explain(req.qid, req.question)
    except Exception as e:
        raise HTTPException(503, f"LLM 호출 실패: {e}")
    if text is None:
        raise HTTPException(404, "없는 문항입니다.")
    return {"qid": req.qid, "explanation": text}

# 이미지 없는 문항 중 모델이 아는 문항 하나를 무작위로 돌려주는 API
@app.get("/random")
def random_question():
    for _ in range(100):
        qid = int(rag.random_id())
        if qid in qmap:
            return {"qid": qid}
    raise HTTPException(404, "문항을 찾지 못했습니다.")

#문항 번호를 주면 문제 본문의 한국어 번역을 돌려주는 API

@app.get("/tutor/translate/{qid}")
def tutor_translate(qid: int):
    ref = rag.get_explanation(qid)
    if ref is None:
        raise HTTPException(404, "없는 문항입니다.")
    try:
        return {"qid": qid, "korean": llm.translate(ref["content"])}
    except Exception as e:
        raise HTTPException(503, f"LLM 호출 실패: {e}")

# 학생의 말과 풀이 기록을 받아 에이전트를 실행하고, 답과 호출한 도구 목록을 돌려주는 API
class ChatRequest(BaseModel):
    message: str
    history: list[Interaction] = []
    current_qid: int | None = None


def _make_tools(history, current_qid):
    def explain_question(qid=None):
        qid = qid or current_qid
        if qid is None:
            return {"error": "문항 번호가 없습니다."}
        return {"explanation": llm.explain(int(qid))}

    def recommend_questions(k=3):
        r = recommend(RecommendRequest(history=history, k=int(k)))
        items, lines = [], []
        for i in r["recommended"]:
            ref = rag.get_explanation(i["qid"])
            ko = llm.translate(ref["content"])
            items.append({"qid": i["qid"], "p_correct": i["p_correct"], "question_ko": ko})
            lines.append(f"- 문항 {i['qid']} (예상 정답 확률 {i['p_correct']:.0%}): {ko}")
        return {"items": items, "summary_ko": "\n".join(lines)}

    def estimate_ability():
        if not history:
            return {"error": "풀이 기록이 없습니다."}
        acc = sum(x.response for x in history) / len(history)
        cands = random.sample(list(qmap), 30)
        res = predict(PredictRequest(history=history, candidates=cands))
        mean_p = sum(p["p_correct"] for p in res["predictions"]) / len(res["predictions"])
        return {
            "n_solved": len(history),
            "accuracy": round(acc, 3),
            "expected_correct_rate": round(mean_p, 3),
            "summary_ko": (
                f"지금까지 {len(history)}문제를 풀었고 정답률은 {acc:.0%}입니다. "
                f"모델이 예측한 무작위 문제의 예상 정답률은 {mean_p:.0%}입니다."
            ),
        }

    return {"explain_question": explain_question,
            "recommend_questions": recommend_questions,
            "estimate_ability": estimate_ability}


@app.post("/tutor/chat")
def tutor_chat(req: ChatRequest):
    ctx = f"(현재 학생이 보고 있는 문항 번호: {req.current_qid}, 지금까지 푼 문제 수: {len(req.history)})"
    try:
        answer, used = agent.run(req.message, _make_tools(req.history, req.current_qid), ctx)
    except Exception as e:
        raise HTTPException(503, f"에이전트 실행 실패: {e}")
    return {"answer": answer, "tools_used": used}

# 문항의 보기를 한국어로 번역해 돌려주는 API
@app.get("/tutor/translate_options/{qid}")
def tutor_translate_options(qid: int):
    ref = rag.get_explanation(qid)
    if ref is None:
        raise HTTPException(404, "없는 문항입니다.")
    opts = ref.get("options")
    if not isinstance(opts, dict) or not opts:
        return {"qid": qid, "options": None}
    try:
        return {"qid": qid, "options": {k: llm.translate_option(str(v)) for k, v in opts.items()}}
    except Exception as e:
        raise HTTPException(503, f"LLM 호출 실패: {e}")