import random

from src import rag



def routes(qid, depth=None):
    r = rag._q[qid].get("kc_routes") or []
    if isinstance(r, str):
        r = [r]
    if depth:
        r = ["----".join(x.split("----")[:depth]) for x in r]
    return set(r)


def evaluate(search_fn, n=300, k=5, seed=0, depth=None):
    random.seed(seed)
    ids = random.sample(rag._ids_all, n)
    used = hit1 = hitk = prec = 0
    for qid in ids:
        mine = routes(qid, depth)
        if not mine:
            continue
        res = search_fn(qid, k)
        if not res:
            continue
        flags = [bool(mine & routes(r["qid"], depth)) for r in res]
        used += 1
        hit1 += flags[0]
        hitk += any(flags)
        prec += sum(flags) / len(flags)
    return {"평가 문항": used, "hit@1": round(hit1 / used, 3),
            f"hit@{k}": round(hitk / used, 3), f"precision@{k}": round(prec / used, 3)}

def random_search(qid, k):
    return [{"qid": c} for c in random.sample(rag._ids_all, k)]



if __name__ == "__main__":
    for d in (None, 3):
        print(f"[경로 깊이 {d or '전체'}]")
        print("  무작위:", evaluate(random_search, depth=d))
        print("  현재 similar():", evaluate(rag.similar, depth=d))
#| 지표          | 무작위   | 현재 similar() |
#| ----------- | ----- | ------------ |
#| hit@1       | 0.003 | 0.423        |
#| hit@5       | 0.020 | 0.677        |
#| precision@5 | 0.004 | 0.317        |