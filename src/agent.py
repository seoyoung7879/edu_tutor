import json

from src.llm import MODEL, client
def _has_han(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)
TOOLS = [
    {"type": "function", "function": {
        "name": "explain_question",
        "description": "특정 문항의 풀이를 한국어로 쉽게 설명한다. qid를 생략하면 학생이 보고 있는 현재 문항을 설명한다.",
        "parameters": {"type": "object", "properties": {
            "qid": {"type": "integer", "description": "문항 번호(선택)"}}}}},
    {"type": "function", "function": {
        "name": "recommend_questions",
        "description": "학생이 틀린 문제와 비슷하고 알맞은 난이도의 다음 문제를 추천한다.",
        "parameters": {"type": "object", "properties": {
            "k": {"type": "integer", "description": "추천 개수(기본 3)"}}}}},
    {"type": "function", "function": {
        "name": "estimate_ability",
        "description": "학생의 지금까지 풀이 기록으로 정답률과 예상 실력을 계산한다.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = (
    "너는 초등 수학 AI 튜터다. 학생의 말에 맞는 도구가 있으면 호출하고, 도구 결과만 근거로 한국어로 답한다. "
    "도구 결과에 없는 내용은 지어내지 않는다. 인사나 잡담에는 도구 없이 짧게 답한다. "
    "중국어로 답하지 않는다."
    "문제를 추천할 때는 문항 번호(qid)를 함께 적는다."
    "도구 결과에 summary_ko가 있으면 그 내용을 그대로 보여 주고 앞뒤에 짧은 안내만 덧붙인다."
)


def run(message, impl, context="", max_steps=4):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{context}\n\n학생 말: {message}"},
    ]
    used = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
            temperature=0.2, max_tokens=700,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = msg.content or ""
            if _has_han(text):
                retry = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "위 답변을 중국어 없이 한국어로만 다시 써 주세요."},
                ]
                r2 = client.chat.completions.create(
                    model=MODEL, messages=retry, temperature=0.2, max_tokens=700
                )
                text = r2.choices[0].message.content or text
            return text, used
        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = impl[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            used.append({"tool": name, "args": args})
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    return "죄송해요, 답을 만들지 못했어요. 다시 말씀해 주세요.", used