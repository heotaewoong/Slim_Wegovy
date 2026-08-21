from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from slim_wegovy.config import load_settings
from slim_wegovy.harness import L2Harness
from slim_wegovy.patient import PatientSimulator


app = FastAPI(title="Slim Wegovy L2 Harness")
_harness_local = threading.local()


def _harness() -> L2Harness:
    harness = getattr(_harness_local, "default", None)
    if harness is None:
        harness = L2Harness(load_settings())
        _harness_local.default = harness
    return harness


def _harness_with_key(api_key: str) -> L2Harness:
    keyed = getattr(_harness_local, "keyed", None)
    if keyed is None:
        keyed = {}
        _harness_local.keyed = keyed
    harness = keyed.get(api_key)
    if harness is None:
        harness = L2Harness(load_settings(api_key_override=api_key))
        keyed[api_key] = harness
    return harness


@lru_cache(maxsize=1)
def _patient() -> PatientSimulator:
    return PatientSimulator(load_settings())


class AskRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)


class PatientRequest(BaseModel):
    history: list[dict[str, str]] = Field(default_factory=list)


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=32_768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.post("/api/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    return _harness().answer(req.question, history=req.history).model_dump()


@app.post("/api/patient")
def patient(req: PatientRequest) -> dict[str, str]:
    return {"question": _patient().next_question(req.history)}


@app.get("/api/tools")
def tools() -> dict[str, Any]:
    return {"tools": _harness()._mcp_tools()}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": _model_name()}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": _model_name(), "object": "model", "created": 0, "owned_by": "lunit"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    if not req.messages:
        return _openai_error("'messages' must be a non-empty array.", 400, "invalid_request_error")
    last_user_index = next(
        (index for index in range(len(req.messages) - 1, -1, -1)
         if req.messages[index].get("role") == "user"),
        None,
    )
    if last_user_index is None:
        return _openai_error("'messages' must contain a user message.", 400, "invalid_request_error")
    question = req.messages[last_user_index].get("content")
    if not isinstance(question, str) or not question.strip():
        return _openai_error("The latest user message must contain text.", 400, "invalid_request_error")

    try:
        result = _request_harness(authorization).answer(
            question,
            history=req.messages[:last_user_index],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except RuntimeError as exc:
        return _openai_error(str(exc), 500, "configuration_error")
    except Exception as exc:
        return _openai_error(f"L2 harness failed: {exc}", 502, "upstream_error")

    completion = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name(),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": getattr(result, "finish_reason", "stop"),
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if req.stream:
        return StreamingResponse(
            _completion_events(completion),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(completion)


def _model_name() -> str:
    return os.environ.get("LUNIT_FM_MODEL", "Lunit/L2-preview")


def _request_harness(authorization: str | None) -> L2Harness:
    try:
        return _harness()
    except RuntimeError as exc:
        token = _bearer_token(authorization)
        if token and "LUNIT_FM_API_KEY" in str(exc):
            return _harness_with_key(token)
        raise


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer":
        return ""
    return token.strip()


def _openai_error(message: str, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


def _completion_events(completion: dict[str, Any]) -> Iterator[str]:
    base = {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
    }
    first = {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    second = {
        **base,
        "choices": [
            {
                "index": 0,
                "delta": {"content": completion["choices"][0]["message"]["content"]},
                "finish_reason": completion["choices"][0]["finish_reason"],
            }
        ],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps(second, ensure_ascii=False)}\n\n"
    usage = {**base, "choices": [], "usage": completion["usage"]}
    yield f"data: {json.dumps(usage, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Slim Wegovy L2 Harness</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #17202a;
      --muted: #697386;
      --brand: #0f766e;
      --accent: #b45309;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 420px;
      height: 100vh;
    }
    main, aside { min-width: 0; }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      height: 36px;
      padding: 0 12px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary { background: var(--brand); border-color: var(--brand); color: #fff; }
    button.warn { color: var(--danger); }
    button:disabled { opacity: .55; cursor: wait; }
    .chat {
      overflow: auto;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: #fbfcfd;
    }
    .msg {
      max-width: 860px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px 14px;
      white-space: pre-wrap;
    }
    .msg.user { align-self: flex-end; background: #eef6f5; border-color: #b7d7d2; }
    .msg.assistant { align-self: flex-start; }
    .msg.pending {
      color: var(--muted);
      border-style: dashed;
    }
    .role {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 6px;
      text-transform: uppercase;
    }
    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 14px 18px;
      border-top: 1px solid var(--line);
      background: #fff;
    }
    textarea {
      width: 100%;
      min-height: 56px;
      max-height: 160px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
    }
    aside {
      display: grid;
      grid-template-rows: auto 1fr;
      background: #f3f5f8;
    }
    .tabs {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      border-bottom: 1px solid var(--line);
    }
    .tab {
      border: 0;
      border-right: 1px solid var(--line);
      border-radius: 0;
      background: #eef1f5;
    }
    .tab.active { background: #fff; color: var(--brand); }
    .panel { overflow: auto; padding: 14px; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
    }
    .event {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 10px;
    }
    .event strong { color: var(--accent); }
    .event code { color: var(--muted); }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) 46vh; }
      main { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <main>
      <header>
        <h1>Slim Wegovy L2 Harness</h1>
        <div class="toolbar">
          <button id="patientBtn" type="button">Patient</button>
          <button id="clearBtn" type="button" class="warn">Clear</button>
        </div>
      </header>
      <section id="chat" class="chat"></section>
      <form id="form">
        <textarea id="question" placeholder="질문을 입력하세요"></textarea>
        <button id="sendBtn" class="primary" type="submit">Send</button>
      </form>
    </main>
    <aside>
      <div class="tabs">
        <button class="tab active" data-tab="trajectory" type="button">Trajectory</button>
        <button class="tab" data-tab="evidence" type="button">Evidence</button>
        <button class="tab" data-tab="raw" type="button">Raw</button>
      </div>
      <div id="panel" class="panel"></div>
    </aside>
  </div>
  <script>
    const chat = document.querySelector("#chat");
    const form = document.querySelector("#form");
    const input = document.querySelector("#question");
    const sendBtn = document.querySelector("#sendBtn");
    const patientBtn = document.querySelector("#patientBtn");
    const clearBtn = document.querySelector("#clearBtn");
    const panel = document.querySelector("#panel");
    const tabs = [...document.querySelectorAll(".tab")];
    let history = [];
    let lastResult = null;
    let activeTab = "trajectory";
    let timerId = null;
    let timerStartedAt = 0;

    function renderChat() {
      chat.innerHTML = "";
      for (const m of history) {
        const el = document.createElement("div");
        el.className = `msg ${m.role}${m.pending ? " pending" : ""}`;
        el.innerHTML = `<div class="role">${m.role}</div>${escapeHtml(m.content)}`;
        chat.appendChild(el);
      }
      chat.scrollTop = chat.scrollHeight;
    }

    function renderPanel() {
      if (!lastResult) {
        panel.innerHTML = "<pre>질문을 보내면 retrieval / generation trajectory가 표시됩니다.</pre>";
        return;
      }
      if (activeTab === "trajectory") {
        panel.innerHTML = (lastResult.tool_events || []).map((e, i) => `
          <div class="event">
            <div><strong>${i + 1}. ${escapeHtml(e.phase)}:${escapeHtml(e.name)}</strong></div>
            <div><code>${escapeHtml(JSON.stringify(e.arguments || {}, null, 2))}</code></div>
            <pre>${escapeHtml((e.result || "").slice(0, 4000))}</pre>
          </div>`).join("") || "<pre>No tool calls.</pre>";
      } else if (activeTab === "evidence") {
        panel.innerHTML = `<pre>${escapeHtml(JSON.stringify({retrieval: lastResult.retrieval, evidence: lastResult.evidence}, null, 2))}</pre>`;
      } else {
        panel.innerHTML = `<pre>${escapeHtml(JSON.stringify(lastResult, null, 2))}</pre>`;
      }
    }

    async function ask(question) {
      setBusy(true);
      history.push({role: "user", content: question});
      history.push({role: "assistant", content: "응답중... 0s", pending: true});
      renderChat();
      startResponseTimer();
      try {
        const prior = history.slice(0, -2);
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question, history: prior})
        });
        if (!res.ok) throw new Error(await res.text());
        lastResult = await res.json();
        replacePendingAssistant(lastResult.answer || "");
      } catch (err) {
        replacePendingAssistant(`오류: ${err.message}`);
      } finally {
        stopResponseTimer();
        renderChat();
        renderPanel();
        setBusy(false);
      }
    }

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const q = input.value.trim();
      if (!q) return;
      input.value = "";
      await ask(q);
    });

    patientBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const res = await fetch("/api/patient", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({history})
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        input.value = data.question || "";
      } catch (err) {
        input.value = `Patient simulator 오류: ${err.message}`;
      } finally {
        setBusy(false);
      }
    });

    clearBtn.addEventListener("click", () => {
      history = [];
      lastResult = null;
      renderChat();
      renderPanel();
    });

    tabs.forEach(tab => tab.addEventListener("click", () => {
      activeTab = tab.dataset.tab;
      tabs.forEach(t => t.classList.toggle("active", t === tab));
      renderPanel();
    }));

    function setBusy(busy) {
      sendBtn.disabled = busy;
      patientBtn.disabled = busy;
    }

    function startResponseTimer() {
      stopResponseTimer();
      timerStartedAt = Date.now();
      timerId = window.setInterval(() => {
        const pending = history.findLast?.(m => m.pending) || [...history].reverse().find(m => m.pending);
        if (!pending) return;
        const seconds = Math.floor((Date.now() - timerStartedAt) / 1000);
        pending.content = `응답중... ${seconds}s`;
        renderChat();
      }, 1000);
    }

    function stopResponseTimer() {
      if (timerId) {
        window.clearInterval(timerId);
        timerId = null;
      }
    }

    function replacePendingAssistant(content) {
      const idx = history.findLastIndex?.(m => m.pending) ?? findLastPendingIndex();
      if (idx >= 0) {
        history[idx] = {role: "assistant", content};
      } else {
        history.push({role: "assistant", content});
      }
    }

    function findLastPendingIndex() {
      for (let i = history.length - 1; i >= 0; i--) {
        if (history[i].pending) return i;
      }
      return -1;
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, s => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[s]));
    }

    renderPanel();
  </script>
</body>
</html>
"""
