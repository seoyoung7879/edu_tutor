# 문제 풀이, 채점, AI 설명, 문제 추천을 한 화면에 묶은 데모 앱
import requests
import streamlit as st

st.set_page_config(page_title="edu_tutor", layout="wide")
API = st.sidebar.text_input("API 주소", "http://127.0.0.1:8000")
ss = st.session_state
for k, v in {"history": [], "q": None, "result": None, "explain": None, "rec": None}.items():
    ss.setdefault(k, v)


def norm(s):
    return "".join(str(s).replace("$", "").split()).lower()


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
    st.markdown(q["content"])
    if q.get("options"):
        st.write(q["options"])
    ans = st.text_input("정답을 입력하세요", key=f"ans_{q['qid']}")
    if st.button("제출") and ans.strip():
        correct = norm(ans) in [norm(a) for a in q["answer"]]
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
        st.write(ss.explain)

    if ss.rec:
        st.markdown("### 추천 문제")
        if "recommended" in ss.rec:
            for item in ss.rec["recommended"]:
                st.markdown(f"**문항 {item['qid']}** (예상 정답 확률 {item['p_correct']:.0%})")
                st.markdown(item["content"])
                st.button("이 문제 풀기", key=f"go_{item['qid']}",
                          on_click=load_question, args=(item["qid"],))
        else:
            st.info(ss.rec.get("detail", "추천 결과가 없습니다."))
#실행: .\.venv-kt\Scripts\python.exe -m streamlit run app\streamlit_app.py