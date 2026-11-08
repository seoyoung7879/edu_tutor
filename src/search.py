import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src import llm, rag

_model = None
_emb = np.load(rag.ROOT / "data/processed/rag_emb.npy")


def _get_model():
    global _model
    if _model is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer("intfloat/multilingual-e5-small", device=dev)
    return _model


def _has_hangul(s):
    return any("\uac00" <= c <= "\ud7a3" for c in s)


def ko_to_zh(text):
    resp = llm.client.chat.completions.create(
        model=llm.MODEL,
        messages=[
            {"role": "system", "content": (
                "한국어 수학 문제나 질문을 중국어(간체)로 번역하라. "
                "숫자와 $$ 수식은 그대로 두고 번역문만 출력하라.")},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def search(text, k=5):
    text = text.strip()
    zh = ko_to_zh(text) if _has_hangul(text) else text
    q = _get_model().encode(["query: " + zh], normalize_embeddings=True)[0]
    top = np.argsort(-(_emb @ q))[:k]
    results = [{"qid": rag._ids_all[i], "score": round(float(_emb[i] @ q), 3),
                "content": rag._q[rag._ids_all[i]]["content"][:200]} for i in top]
    return {"query_zh": zh, "results": results}

def search_for_agent(query, k=3):
    k = max(1, min(int(k or 3), 5))
    q = query.strip()
    out = search_concept(q, k) if len(q) <= 20 else search(q, k)
    items, lines = [], []
    for r in out["results"]:
        content = rag._q[r["qid"]]["content"]
        try:
            ko = llm.translate(content)
        except Exception:
            ko = ""
        if ko.startswith("⚠️"):
            part = ko.split("\n\n", 1)[-1].strip()
            ko = part if part and not llm._has_han(part) else ""
        if not ko:
            ko = "(번역이 불완전해서 요약을 표시하지 못했어요)"
        items.append({"qid": r["qid"], "score": r.get("score"), "summary_ko": ko})
        lines.append(f"- 문항 {r['qid']}번: {ko.replace(chr(10), ' ')[:120]}")
    return {"results": items,
            "summary_ko": "관련 문제를 찾았어요.\n" + "\n".join(lines)}

_concepts = None


def _load_concepts():
    global _concepts
    if _concepts is None:
        by_route = {}
        for qid in rag._ids_all:
            for r in rag._q[qid].get("kc_routes") or []:
                by_route.setdefault(r, []).append(qid)
        routes = list(by_route)
        texts = ["passage: " + r.replace("----", " > ") for r in routes]
        emb = _get_model().encode(texts, batch_size=64, normalize_embeddings=True)
        _concepts = (routes, by_route, emb)
    return _concepts


def search_concept(text, k=3):
    text = text.strip()
    zh = ko_to_zh(text) if _has_hangul(text) else text
    routes, by_route, emb = _load_concepts()
    q = _get_model().encode(["query: " + zh], normalize_embeddings=True)[0]
    best = int(np.argmax(emb @ q))
    cand = sorted(by_route[routes[best]],
                  key=lambda i: -float(_emb[rag._pos_all[i]] @ q))
    out, seen = [], set()
    for i in cand:
        key = rag._key(i)
        if key in seen or "□" in rag._q[i]["content"]:
            continue
        seen.add(key)
        out.append({"qid": i, "score": round(float(_emb[rag._pos_all[i]] @ q), 3),
                    "content": rag._q[i]["content"][:200]})
        if len(out) == k:
            break
    return {"query_zh": zh, "route": routes[best], "results": out}