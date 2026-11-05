# 이미지가 없는 문항의 문제와 해설을 합쳐 임베딩하고 data/processed에 저장
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
questions = json.load(open(ROOT / "data/raw/metadata/questions.json", encoding="utf-8"))
img = re.compile(r"(\.jpg|\.png|image_\d+)", re.I)

ids, texts = [], []
for qid, v in questions.items():
    content = v.get("content") or ""
    analysis = v.get("analysis") or ""
    if not analysis or img.search(content) or img.search(analysis):
        continue
    ids.append(qid)
    texts.append("passage: " + content + "\n" + analysis)
print("전체:", len(questions), "| 이미지 제외 후:", len(ids))

model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

out = ROOT / "data/processed"
np.save(out / "rag_emb.npy", emb.astype("float32"))
json.dump(ids, open(out / "rag_ids.json", "w"))
print("저장 완료:", emb.shape)