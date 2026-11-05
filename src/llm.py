# 문제·정답·공식 해설을 근거로 LLM이 한국어로 풀이를 쉽게 설명하게 하는 함수
import os

from openai import OpenAI

from src import rag

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
    timeout=180,
)
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

SYSTEM = (
    "너는 초등학생을 가르치는 친절한 수학 튜터다. "
    "반드시 아래 [공식 해설]을 근거로만 설명하고, 해설에 없는 내용은 지어내지 마라. "
    "문제와 해설이 중국어이므로 한국어로 옮겨 설명하라. "
    "풀이는 단계별로 짧게, 쉬운 말로 쓰고, 마지막에 정답을 한 줄로 알려 줘라."
)


def explain(qid, student_question=None):
    ref = rag.get_explanation(qid)
    if ref is None:
        return None
    q = (student_question or "").strip()
    if q.lower() in ("", "string"):
        q = "이 문제를 어떻게 푸는지 설명해 주세요."
    user = (
        f"[문제]\n{ref['content']}\n\n[정답]\n{ref['answer']}\n\n"
        f"[공식 해설]\n{ref['analysis']}\n\n[학생 질문]\n{q}\n\n"
        "위 내용을 바탕으로 반드시 한국어로만 답하세요. 중국어로 답하지 마세요. "
        "문제를 먼저 한국어로 한 줄 요약한 뒤 단계별로 설명하세요."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content

# 중국어 문제 본문을 한국어로 번역하고, 같은 문장은 다시 번역하지 않도록 기억
TRANS_MODEL = os.getenv("TRANS_MODEL", "gemma3:4b")
TRANS_SYSTEM = (
    "너는 중국어를 한국어로 옮기는 수학 문제 번역가다. 초등 수학 문제를 자연스러운 한국어로 번역하라. "
    "숫자와 수식은 그대로 두고, 문제를 풀거나 설명을 덧붙이지 말고 한국어 번역문만 출력하라. "
    "중국어 글자를 남기지 마라."
)
_trans_cache = {}


def _han_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum("\u4e00" <= c <= "\u9fff" for c in letters) / len(letters)


def translate(text):
    if text in _trans_cache:
        return _trans_cache[text]
    out = ""
    for _ in range(2):
        resp = client.chat.completions.create(
            model=TRANS_MODEL,
            messages=[
                {"role": "system", "content": TRANS_SYSTEM},
                {"role": "user", "content": "다음 중국어 문제를 한국어로 번역하세요. 번역문만 출력하세요.\n\n" + text},
            ],
            temperature=0,
            max_tokens=400,
        )
        out = resp.choices[0].message.content.strip()
        if _han_ratio(out) < 0.1:
            _trans_cache[text] = out
            return out
    return out