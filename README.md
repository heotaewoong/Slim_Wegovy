# Slim Wegovy

Lunit L2를 대상으로 한 2단계 medical assistant harness baseline입니다.

L2 요청은 기본 generation과 필요한 경우에만 실행하는 retrieval 흐름으로 분리합니다.

- Retrieval trajectory: L2가 MCP tools를 반복 호출해 근거를 모으고 `finalize_retrieval`로 `cite_uid` 목록을 제출합니다.
- Generation trajectory: 안정적인 일반 의학·상담·기본 triage 질문은 바로 답하고, 최신·관할권·문서·허가사항·급여·법률·코드·명시적 출처가 필요한 질문에만 `retrieve_relevant_content`를 한 번 제공합니다.

이 저장소는 CLI와 웹 playground가 같은 `L2Harness`를 사용하도록 구성되어 있습니다.

## Setup

`.env`에는 API key가 필요합니다. 나머지는 아래 값이 기본값입니다.

```bash
LUNIT_FM_API_URL=https://model.hackathon.lunit.io
LUNIT_FM_API_KEY=...
LUNIT_FM_MODEL=Lunit/L2-preview
LUNIT_MCP_URL=https://mcp.hackathon.lunit.io/mcp
MCP_PROTOCOL_VERSION=2025-06-18
```

설치:

```bash
cd /Users/singon/code/Slim_Wegovy
uv sync
```

## CLI

단일 질문:

```bash
uv run slim-wegovy ask "위고비의 주요 이상반응을 알려줘"
```

전체 trajectory JSON까지 출력:

```bash
uv run slim-wegovy ask "위고비 급여 기준이 있나요?" --json
```

Patient simulator로 약 3 turn 대화:

```bash
uv run slim-wegovy simulate --turns 3 --out data/runs/sample.json
```

Simulator의 첫 질문은 수정하지 않고 history에 그대로 보존합니다. 404가 오면 새 대화를 시작하고, 502는 재시도합니다.

## Web Playground

OpenWeb UI처럼 브라우저에서 직접 질문하고 retrieval/generation trajectory를 확인할 수 있습니다.

```bash
uv run slim-wegovy serve --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다.

- `Send`: 직접 입력한 질문을 harness로 실행합니다.
- `Patient`: patient simulator의 다음 질문을 입력창에 가져옵니다.
- `Trajectory`: generation의 `retrieve_relevant_content` 호출과 retrieval MCP tool 호출을 순서대로 표시합니다.
- `Evidence`: `finalize_retrieval` 결과와 선택된 `cite_uid` 근거를 표시합니다.
- `Raw`: 전체 harness 결과를 JSON으로 표시합니다.

## Architecture

주요 파일:

- `slim_wegovy/harness.py`: L2 generation/retrieval loop orchestration
- `slim_wegovy/mcp_client.py`: Lunit Streamable HTTP MCP JSON-RPC client
- `slim_wegovy/tools.py`: `finalize_retrieval`, `retrieve_relevant_content` OpenAI tool schema
- `slim_wegovy/prompts.py`: retrieval/generation system prompts
- `slim_wegovy/patient.py`: OpenAI-compatible patient simulator client
- `slim_wegovy/web.py`: FastAPI web playground
- `slim_wegovy/cli.py`: CLI entrypoint

## Harness Flow

Generation 단계:

1. System prompt와 대화 history, user question을 L2에 전달합니다.
2. 최신·문서 기반 근거가 필요한 질문에만 `retrieve_relevant_content` 하나를 제공합니다.
3. L2가 tool을 호출하면 harness가 retrieval 단계를 실행합니다.
4. retrieval 결과 JSON을 tool result로 되돌려준 뒤에는 tool을 제거하고 L2가 최종 답변을 작성합니다.
5. upstream이 `finish_reason=length`로 문장 중간에 멈추면 한 번의 tool-free continuation으로 남은 요청과 안전 안내를 완결합니다.

Retrieval 단계:

1. L2에 MCP tools 전체와 가상 tool `finalize_retrieval`을 제공합니다.
2. L2가 질문과 관할권에 맞는 작은 MCP tool 집합에서 한 번 근거를 조회합니다.
3. tool result에서 `cite_uid`를 수집합니다.
4. L2가 `finalize_retrieval(status, items, note)`를 호출하면 retrieval phase를 종료합니다.

## Notes

- `.env`는 git에 포함하지 않습니다.
- MCP tool 결과가 길면 `Settings.max_tool_result_chars` 기준으로 잘라 L2 context 폭주를 막습니다.
- 평가 요청의 `max_tokens`와 `temperature`를 내부 L2 호출에 전달합니다. 최종 generation 한도는 CoEval 경쟁 설정과 같은 최대 6,144 tokens입니다.
- generation prompt는 `more than 10 years of experience`인 의료 AI 역할을 명시하되, 이를 권위 주장으로 쓰지 않고 정확성·안전성·근거 정직성의 행동 규칙으로 연결합니다. 영어 질문은 영어로, 한국어 질문은 한국어로 답합니다.
- 요청에 `temperature`가 없으면 재현성을 위해 `0.0`을 사용하고, 명시된 값은 그대로 존중합니다.
- 한 요청은 165초 안에 끝내도록 retrieval에 별도 시간 예산을 두고 최종 답변 시간을 예약합니다.
- 현재 baseline은 citation text를 MCP 결과에서 best-effort로 수집합니다. tool별 결과 shape가 안정적으로 확인되면 `cite_uid` 파싱과 evidence normalization을 더 정교하게 분리하는 것이 다음 개선 지점입니다.

## Hackathon Submission

제출 container는 OpenAI-compatible API를 제공합니다.

- `GET /v1/models`
- `POST /v1/chat/completions`
- 일반 JSON 및 `stream: true` SSE 응답
- multi-turn 전체 history, `max_tokens`, `temperature`, 실제 `finish_reason` 전달
- `GET /health`

로컬 build 및 실행:

```bash
docker build -t slim-wegovy:local .
docker run --rm -p 8000:8000 \
  -e LUNIT_FM_API_KEY="lunit_..." \
  slim-wegovy:local
```

제출값:

- Branch: `lunit/hackathon-submission`
- Server: `0.0.0.0:8000`
- Model: `Lunit/L2-preview`
- Dashboard에는 최종 검증한 branch HEAD의 40자리 전체 SHA를 입력합니다.

API credit을 사용하지 않는 test:

```bash
python -m unittest discover -s tests -v
```
