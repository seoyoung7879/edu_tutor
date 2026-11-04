import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "raw"
QUESTIONS_JSON = BASE / "metadata" / "questions.json"
SEQ_CSV = BASE / "question_level" / "train_valid_sequences_quelevel.csv"
RAG_PARQUET = ROOT / "data" / "processed" / "rag_questions.parquet"

IMG = re.compile(r"question_\d+-image_\d+|analysis_\d+-image_\d+")


def tokens(s):
    return [t for t in str(s).split(",") if t.strip() != ""]


def to_ints(s):
    return [int(t) for t in tokens(s)]


def check_questions(questions):
    results = []
    for field in ["content", "answer", "analysis"]:
        bad = [k for k, v in questions.items() if not str(v.get(field, "")).strip()]
        results.append((f"문항 정보: {field} 비어 있지 않음", len(bad)))
    return results


def check_sequences(seq, known_ids):
    cols = ["questions", "concepts", "responses", "timestamps", "selectmasks"]
    len_bad = resp_bad = ts_bad = id_bad = 0

    for _, row in seq.iterrows():
        lens = {c: len(tokens(row[c])) for c in cols}
        if len(set(lens.values())) != 1:
            len_bad += 1

        responses = to_ints(row["responses"])
        if any(r not in (-1, 0, 1) for r in responses):
            resp_bad += 1

        ts = [t for t in to_ints(row["timestamps"]) if t != -1]
        if any(b < a for a, b in zip(ts, ts[1:])):
            ts_bad += 1

        qs = {q for q in to_ints(row["questions"]) if q != -1}
        if not qs <= known_ids:
            id_bad += 1

    return [
        ("풀이 로그: 5개 칸의 길이가 서로 같음", len_bad),
        ("풀이 로그: responses가 -1/0/1만 포함", resp_bad),
        ("풀이 로그: timestamps가 시간순", ts_bad),
        ("풀이 로그: 모든 문항 번호가 문항 정보에 존재", id_bad),
    ]


def check_rag(rag_path, known_ids):
    if not rag_path.exists():
        return [("RAG 파일 존재", 1)]
    df = pd.read_parquet(rag_path)
    results = []
    results.append(("RAG: 본문 이미지 표시 없음", int(df["content"].map(lambda x: bool(IMG.search(x))).sum())))
    results.append(("RAG: 해설 이미지 표시 없음", int(df["analysis"].map(lambda x: bool(IMG.search(x))).sum())))
    results.append(("RAG: 문항 번호가 중복 없음", int(df["qid"].duplicated().sum())))
    results.append(("RAG: 모든 문항 번호가 문항 정보에 존재", int((~df["qid"].isin(known_ids)).sum())))
    return results


def main():
    with open(QUESTIONS_JSON, encoding="utf-8") as f:
        questions = json.load(f)
    known_ids = {int(k) for k in questions}
    seq = pd.read_csv(SEQ_CSV)

    results = []
    results += check_questions(questions)
    results += check_sequences(seq, known_ids)
    results += check_rag(RAG_PARQUET, known_ids)

    failed = 0
    for name, n_bad in results:
        status = "통과" if n_bad == 0 else f"실패 {n_bad}건"
        print(f"[{status}] {name}")
        if n_bad:
            failed += 1

    print(f"\n통과 {len(results) - failed}개 / 실패 {failed}개")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()