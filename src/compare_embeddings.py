# 지금 쓰는 임베딩(기준선)과 본문만, 해설만, 질문 접두어 방식을 같은 기준으로 비교합니다.
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import rag
from src.eval_rag import evaluate

import torch

model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

def encode(textfn, prefix):
    texts = [prefix + textfn(rag._q[q]) for q in rag._ids_all]
    return model.encode(texts, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=True).astype("float32")


def make_search(emb):
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    def search(qid, k):
        qid = str(qid)
        i = rag._pos_all[qid]
        scores, ids = idx.search(emb[i:i + 1], 60)
        me, seen, out = rag._key(qid), set(), []
        for s, j in zip(scores[0], ids[0]):
            cand = rag._ids_all[j]
            if "□" in rag._q[cand]["content"]:
                continue
            key = rag._key(cand)
            if cand == qid or key == me or key in seen:
                continue
            seen.add(key)
            out.append({"qid": cand, "score": float(s)})
            if len(out) == k:
                break
        return out
    return search


if __name__ == "__main__":
    variants = {
        "현재 rag_emb.npy": np.load(rag.ROOT / "data/processed/rag_emb.npy"),
        "A 본문만": encode(lambda v: v["content"], "passage: "),
        "C 해설만": encode(lambda v: v["analysis"], "passage: "),
        "D 본문+해설, query 접두어": encode(
            lambda v: v["content"] + "\n" + v["analysis"], "query: "),
    }
    for name, emb in variants.items():
        for d in (None, 3):
            print(name, "| 깊이", d or "전체", evaluate(make_search(emb), depth=d))
#| 방식                 | hit@1 (전체/3단계) | hit@5 (전체/3단계) | precision@5 (전체/3단계) |
#| ------------------ | -------------- | -------------- | -------------------- |
#| 현재 rag_emb.npy     | 0.560 / 0.773  | 0.760 / 0.893  | 0.373 / 0.675        |
#| A 본문만              | 0.540 / 0.760  | 0.723 / 0.893  | 0.357 / 0.656        |
#| C 해설만              | 0.517 / 0.757  | 0.710 / 0.853  | 0.331 / 0.622        |
#| D 본문+해설, query 접두어 | 0.577 / 0.793  | 0.757 / 0.900  | 0.377 / 0.686        |