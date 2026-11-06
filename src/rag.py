# 해설 꺼내기와 비슷한 문제 찾기 함수입니다. 같은 문제는 결과에서 빼 줍니다.
import json
from pathlib import Path

import numpy as np
import random

ROOT = Path(__file__).resolve().parent.parent
_q = json.load(open(ROOT / "data/raw/metadata/questions.json", encoding="utf-8"))
_emb_all = np.load(ROOT / "data/processed/rag_emb.npy")
_ids_all = json.load(open(ROOT / "data/processed/rag_ids.json"))
_pos_all = {qid: i for i, qid in enumerate(_ids_all)}


def _key(qid):
    v = _q[qid]
    return (v["content"].strip(), v["analysis"].strip())

# 해설 조회에 문제 유형과 보기를 포함하고, 랜덤 문항을 고르는 함수를 추가
def get_explanation(qid):
    v = _q.get(str(qid))
    if v is None:
        return None
    return {
        "qid": str(qid), "content": v["content"], "answer": v["answer"],
        "analysis": v["analysis"], "type": v.get("type"), "options": v.get("options"),
    }


def random_id():
    for _ in range(100):
        qid = random.choice(_ids_all)
        if "□" not in _q[qid]["content"]:
            return qid
    return random.choice(_ids_all)

def similar(qid, k=3):
    qid = str(qid)
    if qid not in _pos_all:
        return None
    scores = _emb_all @ _emb_all[_pos_all[qid]]
    me, seen, out = _key(qid), set(), []
    for i in np.argsort(-scores):
        cand = _ids_all[i]
        if "□" in _q[cand]["content"]:
            continue
        key = _key(cand)
        #점수가 0.98 이상인 결과는 같은 문제로 보고 건너뛰기
        if cand == qid or key == me or key in seen or scores[i] >= 0.98:
            continue
        seen.add(key)
        out.append({"qid": cand, "score": round(float(scores[i]), 3), "content": _q[cand]["content"][:200]})
        if len(out) == k:
            break
    return out