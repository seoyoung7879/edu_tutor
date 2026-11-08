import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src import llm, rag
from src.eval_rag import routes

DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("intfloat/multilingual-e5-small", device=DEV)
emb = np.load(rag.ROOT / "data/processed/rag_emb.npy")


def search_text(text, k=5):
    q = model.encode(["query: " + text], normalize_embeddings=True)[0]
    top = np.argsort(-(emb @ q))[:k]
    return [rag._ids_all[i] for i in top]


def main(n=100, k=5, seed=0):
    random.seed(seed)
    ids = random.sample(rag._ids_all, n)
    used = 0
    m = {"ko": [0, 0], "ko→zh": [0, 0], "zh": [0, 0]}
    for qid in ids:
        zh = rag._q[qid]["content"]
        ko = llm.translate(zh)
        if ko.startswith("⚠️"):
            continue
        mine = routes(qid, 3)
        used += 1
        for name, text in (("ko", ko), ("ko→zh", ko_to_zh(ko)), ("zh", zh)):
            res = search_text(text, k)
            m[name][0] += qid in res
            m[name][1] += any(mine & routes(r, 3) for r in res)
    print({"평가 문항": used})
    for name, (e, c) in m.items():
        print(name, {"원문 적중": round(e / used, 3), "단원 적중": round(c / used, 3)})
def ko_to_zh(text):
    resp = llm.client.chat.completions.create(
        model=llm.MODEL,
        messages=[
            {"role": "system", "content": (
                "한국어 수학 문제를 중국어(간체)로 번역하라. "
                "숫자와 $$ 수식은 그대로 두고 번역문만 출력하라.")},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    main()
#{'평가 문항': 78, '한국어 질문 원문 적중': 0.628, '한국어 질문 단원 적중': 0.859, '중국어 질문 원문 적중': 0.974, '중국어 질문 단원 적중': 1.0}
# {'평가 문항': 78} ko {'원문 적중': 0.654, '단원 적중': 0.872} ko→zh {'원문 적중': 0.872, '단원 적중': 0.949} zh {'원문 적중': 0.974, '단원 적중': 1.0}
#한국어 질문을 중국어로 바꿔서 검색하는 방식이 효과가 확실