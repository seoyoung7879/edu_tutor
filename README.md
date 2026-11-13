# 초등 수학 AI 튜터

학생의 풀이 기록으로 실력을 파악하고, 공식 해설을 근거로 한국어 풀이를 설명하며, 다음 문제를 추천하는 학습 도우미입니다. AWS EC2에 Docker로 배포해 인터넷에서 쓸 수 있습니다.

## 주요 기능
- **DKT(지식 추적)**: 풀이 이력으로 문항별 정답 확률을 예측합니다.
- **RAG 검색**: 문제·해설 임베딩(multilingual-e5-small) 검색으로 관련 문제를 찾습니다.
- **추천**: 틀린 문제와 비슷한 후보 중 예측 정답 확률이 0.6에 가까운 문제를 골라 줍니다.
- **LLM 튜터 에이전트**: 도구 4개(`explain_question`, `recommend_questions`, `estimate_ability`, `search_questions`)를 호출해 한국어로 답합니다.
- **번역**: 중국어 문제를 한국어로 번역합니다. 숫자 보존 검사와 재시도, 원문 병기를 하며, 사전 번역 캐시 4,045건을 포함합니다.
- **기록 저장**: 학생 ID별 풀이 기록을 DynamoDB에 저장합니다.
- **화면**: Streamlit (문제 풀이, 채점, AI 설명, 추천, 튜터와 대화)

## 구성
- 실행: AWS EC2(t3.small)에서 Docker Compose로 `api`(FastAPI, 8000)와 `ui`(Streamlit, 8501) 실행
- LLM: OpenAI 호환 API (로컬 Ollama 또는 Gemini API)
- 저장: DynamoDB, 모델·데이터 전달은 S3
- 권한: EC2 IAM 역할 사용 (코드에 키 없음)
- CI: GitHub Actions가 이미지를 빌드하고 컨테이너를 띄워 헬스체크

## 서빙 수치 (EC2, 유휴 상태)
| 항목 | 값 |
| --- | --- |
| API 이미지 | 디스크 2.97GB, 압축 776MB |
| UI 이미지 | 디스크 841MB, 압축 193MB |
| 메모리 | API 약 454MiB, UI 약 66MiB |

## 주요 API
- `POST /tutor/chat`: 에이전트 대화 (`student_id`를 보내면 저장된 기록 사용)
- `POST /tutor/explain`: 공식 해설 기반 한국어 풀이
- `POST /recommend`, `POST /predict`: 추천, 정답 확률 예측
- `POST /students/{id}/answers`, `GET /students/{id}/history`: 풀이 기록 저장·조회
- `GET /health`

## 알려진 한계
- 번역과 설명은 LLM이 만들어서 틀릴 수 있습니다. 화면에 원문과 경고를 함께 둡니다.
- 본문이 같은 문항이 있어 검색 결과가 요청 개수보다 적게 나올 수 있습니다.
