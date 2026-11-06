# 문제·정답·공식 해설을 근거로 LLM이 한국어로 풀이를 쉽게 설명하게 하는 함수
import os

from openai import OpenAI

from src import rag
import re

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
    "공식 해설의 계산과 결론을 그대로 따르고, 새로운 풀이나 다른 계산을 만들지 마라"
)


def explain(qid, student_question=None):
    ref = rag.get_explanation(qid)
    if ref is None:
        return None
    try:
        ko = translate(ref["content"])
    except Exception:
        ko = "(번역 없음)"
    q = (student_question or "").strip()
    if q.lower() in ("", "string"):
        q = "이 문제를 어떻게 푸는지 설명해 주세요."
    user = (
        f"[문제(중국어 원문)]\n{ref['content']}\n\n[문제(한국어 번역)]\n{ko}\n\n"
        f"[정답]\n{ref['answer']}\n\n[공식 해설]\n{ref['analysis']}\n\n[학생 질문]\n{q}\n\n"
        "위 내용을 바탕으로 반드시 한국어로만 답하세요. 중국어로 답하지 마세요. "
        "문제에 나오는 물건과 사람은 [문제(한국어 번역)]에 적힌 표현을 그대로 쓰세요. "
        "단계별로 짧고 쉽게 설명하고 마지막에 정답을 알려 주세요."
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
    "2) 숫자와 수식은 그대로 두고, 수식을 감싼 $$ 기호를 절대 지우지 않는다. "
    "3) 문제를 풀거나 설명을 덧붙이지 않는다. 문장을 요약하거나 생략하지 않는다. "
    "4) 원문에 （ ）나 □ 같은 빈칸 기호가 있으면 그대로 ( ), □로 둔다. "
    "빈칸 기호가 없이 '…有种方法', '…做对了题'처럼 수량이 빠진 문장만 "
    "'몇 개(몇 가지)인지 구하시오' 형태의 질문으로 번역한다. "
    "5) 不做或做错=풀지 않았거나 틀린 경우, 倒扣=감점, 小红=샤오훙, 艾迪=에디, "
    "堆=무더기(덤불 아님), 分给=나누어 주다."
)
PLAIN_SYSTEM = (
    "너는 중국어를 한국어로 옮기는 번역가다. 입력 문장을 요약하거나 생략하지 말고 빠짐없이 그대로 번역하라. "
    "숫자, 수식($$ 포함)은 그대로 두고, 중국어 사람 이름은 중국어 발음으로 적는다. "
    "'是岁', '做对了题'처럼 값이 지워진 자리는 ( )로 표시한다. "
    "한국어 번역문만 출력하고 풀이는 하지 않는다. 한자를 남기지 마라."
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



def _chat(messages, temp=0.0, max_tokens=500):
    resp = client.chat.completions.create(
        model=TRANS_MODEL, messages=messages, temperature=temp, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()


def _nums(s):
    return set(re.findall(r"\d+(?:\.\d+)?", s)) - {"1", "2"}


def _good(src, out):
    return bool(out) and not _has_han(out) and _nums(src) <= _nums(out)


def translate(text):
    if text in _trans_cache:
        return _trans_cache[text]
    for temp in (0, 0.3):
        out = _chat(
            [{"role": "system", "content": TRANS_SYSTEM}, *_FEWSHOT,
             {"role": "user", "content": _tr_user(text)}], temp)
        if _good(text, out):
            _trans_cache[text] = out
            return out
    chunks = [c for c in re.split(r"(?<=[；;。．！？!?])", text) if c.strip()]
    out2 = " ".join(
        _chat([{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": c}])
        for c in chunks
    )
    if _good(text, out2):
        _trans_cache[text] = out2
        return out2
    return "⚠️ 번역이 불완전할 수 있어요. 아래 중국어 원문을 함께 확인하세요.\n\n" + out2
# 보기 문구 하나를 한국어로 번역하고, 한자가 남으면 한 번 더 시도
_opt_cache = {}


_CN = {"零": "0", "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
       "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}


def translate_option(text):
    if text in _opt_cache:
        return _opt_cache[text]
    core = re.sub(r"[\$\s]", "", text)
    if not core:
        return ""
    if core in _CN:
        return _CN[core]
    if not _has_han(text):
        return text
    out = text
    for temp in (0, 0.3):
        resp = client.chat.completions.create(
            model=TRANS_MODEL,
            messages=[
                {"role": "system", "content": (
                    "수학 문제의 보기 문구를 한국어로 번역하라. 숫자와 $$ 수식은 그대로 두고 "
                    "한국어 번역문만 짧게 출력하라. 한자를 남기지 마라. "
                    "个=개, 种=가지, 岁=살, 元=원, 厘米=cm, 米=m.")},
                {"role": "user", "content": text},
            ],
            temperature=temp,
            max_tokens=100,
        )
        out = resp.choices[0].message.content.strip()
        if "죄송" in out or "번역할" in out:
            return text
        if not _has_han(out):
            _opt_cache[text] = out
            break
    return out