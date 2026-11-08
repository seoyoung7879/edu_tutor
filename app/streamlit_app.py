# 문제 풀이, 채점, AI 설명, 문제 추천을 한 화면에 묶은 데모 앱
import re

import requests
import streamlit as st

st.set_page_config(page_title="edu_tutor", layout="wide")
API = st.sidebar.text_input("API 주소", "http://127.0.0.1:8000")
ss = st.session_state
for k, v in {"history": [], "q": None, "result": None, "explain": None, "rec": None, "trans": {}}.items():
    ss.setdefault(k, v)


def norm(s):
    return "".join(str(s).replace("$", "").split()).lower()
# 입력을 쉼표·공백·、 기준으로 나눠 칸별로 비교하고, 칸이 하나뿐이면 전체 문자열 비교도 허용
def split_ans(s):
    return [norm(x) for x in re.split(r"[,，、;；\s]+", s.strip()) if x]


def is_correct(ans, answers):
    answers_n = [norm(a) for a in answers]
    if len(answers_n) > 1:
        return split_ans(ans) == answers_n
    return norm(ans) in answers_n or split_ans(ans) == answers_n


_SYM = {"div": "÷", "times": "×", "cdot": "·", "leq": "≤", "le": "≤",
        "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠"}


def fix_math(text):
    t = (text or "").replace("$$", "$")
    t = re.sub(r"\\\((.+?)\\\)", r"$\1$", t, flags=re.S)
    t = re.sub(r"\\\[(.+?)\\\]", r"$\1$", t, flags=re.S)
    t = re.sub(r"\\(div|times|cdot|leq|geq|neq|le|ge|ne)(?![a-zA-Z])",
               lambda m: _SYM[m.group(1)], t)
    parts = t.split("$")
    if len(parts) % 2 == 1:
        pat = r"\\(?:frac(?:\{[^{}]*\}){2}|sqrt\{[^{}]*\})"
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(pat, lambda m: "$" + m.group(0) + "$", parts[i])
        t = "$".join(parts)
    return t


def ko_text(qid, fallback):
    qid = int(qid)
    if qid in ss.trans:
        return ss.trans[qid]
    try:
        r = requests.get(f"{API}/tutor/translate/{qid}", timeout=180)
        if r.ok and r.json().get("korean"):
            ss.trans[qid] = r.json()["korean"]
            return ss.trans[qid]
        st.warning(f"문항 {qid} 번역 실패: {r.status_code} {r.text[:200]}")
    except requests.RequestException as e:
        st.warning(f"문항 {qid} 번역 요청 오류: {e}")
    return fallback


def load_question(qid):
    ss.q = requests.get(f"{API}/explain/{qid}", timeout=10).json()
    ss.result = ss.explain = ss.rec = None


def new_question():
    qid = requests.get(f"{API}/random", timeout=10).json()["qid"]
    load_question(qid)


with st.sidebar:
    n = len(ss.history)
    if n:
        acc = sum(h["response"] for h in ss.history) / n
        st.metric("정답률", f"{acc:.0%}", f"{n}문제 풀이")
    if st.button("기록 초기화"):
        ss.history = []

st.title("edu_tutor: AI 학습 도우미")

try:
    if st.button("새 문제"):
        new_question()
except requests.RequestException as e:
    st.error(f"API 서버에 연결할 수 없습니다: {e}")

q = ss.q
if q:
    st.subheader(f"문항 {q['qid']}")
    with st.spinner("한국어로 번역하는 중..."):
        ko = ko_text(q["qid"], "(번역 실패 — 아래 원문을 확인하세요)")
    st.markdown(fix_math(ko))
    with st.expander("중국어 원문 보기"):
        st.markdown(fix_math(q["content"]))
    ans = None
    if q.get("options"):
        okey = f"optko_{q['qid']}"
        if okey not in ss:
            try:
                r = requests.get(f"{API}/tutor/translate_options/{q['qid']}", timeout=120)
                ss[okey] = r.json().get("options") or q["options"]
            except requests.RequestException:
                ss[okey] = q["options"]
        opts_ko = ss[okey]
        ans = st.radio(
            "보기를 고르세요", list(q["options"].keys()), index=None, key=f"radio_{q['qid']}",
            format_func=lambda k: f"{k}. {fix_math(str(opts_ko.get(k, q['options'][k])))}",
        )
    else:
        if len(q["answer"]) > 1:
            st.caption(f"정답이 {len(q['answer'])}개 입니다. 쉼표로 구분해 순서대로 입력하세요.")
        ans = st.text_input("정답을 입력하세요", key=f"ans_{q['qid']}")

    if st.button("제출") and ans and str(ans).strip():
        if q.get("options"):
            chosen = [str(ans), str(q["options"][str(ans)])]
            correct = any(is_correct(c, q["answer"]) for c in chosen)
        else:
            correct = is_correct(str(ans), q["answer"])
        ss.history.append({"qid": int(q["qid"]), "response": int(correct)})
        ss.result = correct
        ss.explain = ss.rec = None

    if ss.result is not None:
        if ss.result:
            st.success("정답입니다!")
        else:
            st.error("오답입니다. 정답: " + ", ".join(q["answer"]))
        c1, c2 = st.columns(2)
        if c1.button("AI 설명 듣기"):
            with st.spinner("설명을 만드는 중... (시간이 걸릴 수 있어요)"):
                r = requests.post(f"{API}/tutor/explain", json={"qid": int(q["qid"])}, timeout=300)
                ss.explain = r.json().get("explanation", r.text)
        if c2.button("비슷한 문제 추천"):
            r = requests.post(f"{API}/recommend", json={"history": ss.history, "k": 3}, timeout=60)
            ss.rec = r.json()

    if ss.explain:
        st.markdown("### AI 설명")
        st.markdown(fix_math(ss.explain))

    if ss.rec:
        st.markdown("### 추천 문제")
        if "recommended" in ss.rec:
            with st.spinner("추천 문제를 번역하는 중..."):
                texts = {i["qid"]: ko_text(i["qid"], i["content"]) for i in ss.rec["recommended"]}
            for item in ss.rec["recommended"]:
                st.markdown(f"**문항 {item['qid']}** (예상 정답 확률 {item['p_correct']:.0%})")
                st.markdown(fix_math(texts[item["qid"]]))
                st.button("이 문제 풀기", key=f"go_{item['qid']}",
                          on_click=load_question, args=(item["qid"],))
        else:
            st.info(ss.rec.get("detail", "추천 결과가 없습니다."))
# 풀이 기록과 현재 문항을 에이전트 API에 보내고, 답변과 사용한 도구를 대화 형태로 보여 주는 UI
st.divider()
st.markdown("### 튜터와 대화하기")
ss.setdefault("chat", [])
for m in ss.chat:
    with st.chat_message(m["role"]):
        st.markdown(fix_math(m["content"]))
        if m.get("tools"):
            st.caption("사용한 도구: " + ", ".join(m["tools"]))

if prompt := st.chat_input("튜터에게 물어보세요 (예: 내 실력이 어때?)"):
    ss.chat.append({"role": "user", "content": prompt})
    with st.spinner("튜터가 생각하는 중..."):
        try:
            r = requests.post(
                f"{API}/tutor/chat",
                json={"message": prompt, "history": ss.history,
                      "current_qid": int(ss.q["qid"]) if ss.q else None},
                timeout=300,
            )
            data = r.json()
            ans = data.get("answer") or str(data)
            tools = [t["tool"] for t in data.get("tools_used", [])]
        except requests.RequestException as e:
            ans, tools = f"요청 오류: {e}", []
    ss.chat.append({"role": "assistant", "content": ans, "tools": tools})
    st.rerun()
#api: .\.venv-kt\Scripts\python.exe -m uvicorn src.serve:app --reload
# http://127.0.0.1:8000/docs
#실행: .\.venv-kt\Scripts\python.exe -m streamlit run app\streamlit_app.py