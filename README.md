# Slim Wegovy

Lunit L2를 대상으로 한 2단계 medical assistant harness baseline입니다.

L2는 범용 chat model처럼 한 번 호출해서 끝내는 모델이 아니라, 다음 두 흐름을 분리해서 제어해야 합니다.

- Retrieval trajectory: L2가 MCP tools를 반복 호출해 근거를 모으고 `finalize_retrieval`로 `cite_uid` 목록을 제출합니다.
- Generation trajectory: L2가 답변 작성 중 추가 근거가 필요하다고 판단하면 `retrieve_relevant_content` 하나만 호출하고, harness가 내부적으로 retrieval trajectory를 실행합니다.

이 저장소는 CLI와 웹 playground가 같은 `L2Harness`를 사용하도록 구성되어 있습니다.

## Setup

`.env`에 다음 값이 필요합니다.

```bash
LUNIT_FM_API_URL=https://model.hackathon.lunit.io
LUNIT_FM_API_KEY=...
LUNIT_FM_MODEL=Lunit/L2-preview
LUNIT_MCP_URL=https://mcp.hackathon.lunit.io/mcp
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

만약 껐다 켜야하면
```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <pid>
```

- `Send`: 직접 입력한 질문을 harness로 실행합니다.
- `Patient`: patient simulator의 다음 질문을 입력창에 가져옵니다.
- `Trajectory`: generation의 `retrieve_relevant_content` 호출과 retrieval MCP tool 호출을 순서대로 표시합니다.
- `Evidence`: `finalize_retrieval` 결과와 선택된 `cite_uid` 근거를 표시합니다.
- `Raw`: 전체 harness 결과를 JSON으로 표시합니다.

대화는 multi-turn으로 동작합니다. 사용자가 후속 질문을 보내면 이전 user/assistant turn 전체가 다음 `/api/ask` 요청에 포함되고, 서버 응답의 `history`로 브라우저 상태를 갱신합니다. 브라우저 새로고침 후에도 이어서 실험할 수 있도록 `localStorage`에 history를 저장합니다. `Clear`를 누르면 저장된 history도 함께 삭제됩니다.

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
2. 제공 tool은 `retrieve_relevant_content` 하나뿐입니다.
3. L2가 tool을 호출하면 harness가 retrieval 단계를 실행합니다.
4. retrieval 결과 JSON을 tool result로 되돌려주고 L2가 최종 답변을 작성합니다.

Retrieval 단계:

1. L2에 MCP tools 전체와 가상 tool `finalize_retrieval`을 제공합니다.
2. L2가 MCP tools로 검색, 열람, 구조 탐색, 약가/허가/법령/가이드라인 조회 등을 수행합니다.
3. tool result에서 `cite_uid`를 수집합니다.
4. L2가 `finalize_retrieval(status, items, note)`를 호출하면 retrieval phase를 종료합니다.

## Notes

- `.env`는 git에 포함하지 않습니다.
- MCP tool 결과가 길면 `Settings.max_tool_result_chars` 기준으로 잘라 L2 context 폭주를 막습니다.
- 현재 baseline은 citation text를 MCP 결과에서 best-effort로 수집합니다. tool별 결과 shape가 안정적으로 확인되면 `cite_uid` 파싱과 evidence normalization을 더 정교하게 분리하는 것이 다음 개선 지점입니다.

## Evaluator 제출

제출용 서비스는 repository root의 `Dockerfile`로 빌드하며 `0.0.0.0:8000`에서 다음 OpenAI-compatible endpoint를 제공합니다.

- `GET /v1/models`
- `POST /v1/chat/completions`

`chat/completions`는 evaluator가 전달한 마지막 `user` message를 현재 질문으로 처리하고, 그 앞의 user/assistant messages를 multi-turn history로 L2에 전달합니다. `.env`는 이미지에 포함하지 않으며 컨테이너 실행 시 필요한 환경변수를 주입합니다.

```bash
docker build -t slim-wegovy-submission:local .
docker run --rm -p 8000:8000 \
  -e LUNIT_FM_API_URL="$LUNIT_FM_API_URL" \
  -e LUNIT_FM_API_KEY="$LUNIT_FM_API_KEY" \
  -e LUNIT_FM_MODEL="$LUNIT_FM_MODEL" \
  -e LUNIT_MCP_URL="${LUNIT_MCP_URL:-https://mcp.hackathon.lunit.io/mcp}" \
  slim-wegovy-submission:local
```

제출 branch는 `lunit/hackathon-submission`이며, 제출 화면에는 이 branch의 `HEAD` 전체 SHA와 `LUNIT_FM_MODEL` 값을 입력합니다.
