import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
questions = json.load(open(ROOT / "data/raw/metadata/questions.json", encoding="utf-8"))
emb = np.load(ROOT / "data/processed/rag_emb.npy")
ids = json.load(open(ROOT / "data/processed/rag_ids.json"))
model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
print("불러오기 완료:", emb.shape, flush=True)

print("\n[1] 한국어 질문 검색 (눈으로 확인)", flush=True)
for q in ["경우의 수 순열 문제", "도형의 넓이 구하기", "방정식 풀이"]:
    qv = model.encode(["query: " + q], normalize_embeddings=True)[0]
    scores = emb @ qv
    print("\n질문:", q, flush=True)
    for i in np.argsort(-scores)[:3]:
        text = questions[ids[i]]["content"][:60].replace("\n", " ")
        print(" ", round(float(scores[i]), 3), ids[i], text, flush=True)

print("\n[2] 자기 문제로 검색 (숫자로 확인)", flush=True)
random.seed(0)
sample = random.sample(range(len(ids)), 200)
qtexts = ["query: " + questions[ids[i]]["content"][:200] for i in sample]
qv = model.encode(qtexts, batch_size=64, normalize_embeddings=True)
top = np.argsort(-(qv @ emb.T), axis=1)[:, :5]
hit1 = np.mean([sample[k] == top[k][0] for k in range(200)])
hit5 = np.mean([sample[k] in top[k] for k in range(200)])
print("hit@1:", round(float(hit1), 3), "| hit@5:", round(float(hit5), 3), flush=True)