import json
from pathlib import Path

from src import llm, rag

ROOT = Path(__file__).resolve().parents[1]
IDS = ROOT / "data" / "processed" / "rag_ids.json"
OUT = ROOT / "data" / "processed" / "ko_cache.json"
FAILED = ROOT / "data" / "processed" / "ko_failed.json"


def save(cache, failed):
    OUT.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    FAILED.write_text(json.dumps(failed), encoding="utf-8")


def main():
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    failed = []
    for n, qid in enumerate(ids, 1):
        content = rag.get_explanation(qid)["content"]
        if content in cache:
            continue
        try:
            ko = llm.translate(content)
        except Exception as e:
            print("오류", qid, e)
            failed.append(qid)
            continue
        if ko.startswith("⚠️"):
            failed.append(qid)
            continue
        cache[content] = ko
        if n % 25 == 0:
            save(cache, failed)
            print(f"{n}/{len(ids)} | 저장 {len(cache)} | 실패 {len(failed)}", flush=True)
    save(cache, failed)
    print("완료:", len(cache), "개 저장, 실패", len(failed), "개")


if __name__ == "__main__":
    main()