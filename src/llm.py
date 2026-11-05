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

_trans_cache = {}
TRANS_SYSTEM = (
    "너는 중국어 초등 수학 문제를 한국어로 옮기는 번역가다. 규칙을 반드시 지켜라. "
    "1) 한국어 번역문만 출력한다. 중국어 글자(한자)를 하나도 남기지 않는다. "
    "2) 숫자와 $$ 수식은 그대로 둔다. "
    "3) 문제를 풀거나 설명을 덧붙이지 않는다. "
    "4) 원문은 빈칸이 지워져 있다. '…有种方法', '…做对了题'처럼 수량이 빠진 문장은 "
    "단정문이 아니라 '몇 개(몇 가지)인지 구하시오' 형태의 질문으로 번역한다. "
    "5) 不做或做错=풀지 않았거나 틀린 경우, 倒扣=감점, 小红=샤오훙, 艾迪=에디."
)


def _tr_user(t):
    return "다음 중국어 문제를 한국어로 번역하세요. 번역문만 출력하세요.\n\n" + t


_FEWSHOT = [
    {"role": "user", "content": _tr_user("将$$8$$拆成$$3$$个互不相同的自然数的和，共有种不同的方法．")},
    {"role": "assistant", "content": "$$8$$을 서로 다른 자연수 $$3$$개의 합으로 나타내는 방법은 모두 몇 가지인지 구하시오."},
    {"role": "user", "content": _tr_user("某次数学竞赛共$$15$$道题，每做对一题得$$8$$分，不做或做错一题倒扣$$5$$分，小红得了$$81$$分，小红做对了题．")},
    {"role": "assistant", "content": "어느 수학 경시대회에 문제가 모두 $$15$$개 있었다. 한 문제를 맞힐 때마다 $$8$$점을 얻고, 풀지 않았거나 틀리면 한 문제마다 $$5$$점을 깎는다. 샤오훙이 $$81$$점을 얻었다면 맞힌 문제는 몇 개인지 구하시오."},
]


def _has_han(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def translate(text):
    if text in _trans_cache:
        return _trans_cache[text]
    out = ""
    for temp in (0, 0.3, 0.3):
        resp = client.chat.completions.create(
            model=TRANS_MODEL,
            messages=[{"role": "system", "content": TRANS_SYSTEM}, *_FEWSHOT,
                      {"role": "user", "content": _tr_user(text)}],
            temperature=temp,
            max_tokens=400,
        )
        out = resp.choices[0].message.content.strip()
        if not _has_han(out):
            _trans_cache[text] = out
            return out
    return out