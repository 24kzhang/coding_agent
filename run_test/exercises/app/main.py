"""FastAPI 应用入口：页面、数据、媒体与聊天流式接口。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import chat
import data

BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE / "tpl"))

app = FastAPI(title="健身动作库")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)
    exercise_id: str = Field(min_length=1, max_length=20)
    message: str = Field(min_length=1, max_length=chat.MAX_MESSAGE_CHARS)


class ResetIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "服务暂时不可用，请稍后重试"})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"facets": data.facets(), "total": data.TOTAL},
    )


@app.get("/api/exercises")
async def list_exercises(
    category: str = "", target: str = "", equipment: str = "", q: str = ""
) -> dict[str, Any]:
    items = data.filter_exercises(category=category, target=target, equipment=equipment, q=q)
    return {
        "total": data.TOTAL,
        "count": len(items),
        "items": [data.summary(item) for item in items],
    }


@app.get("/api/exercises/{exercise_id}")
async def exercise_detail(exercise_id: str) -> dict[str, Any]:
    item = data.get_exercise(exercise_id)
    if item is None:
        raise HTTPException(status_code=404, detail="未找到该动作")
    return item


@app.get("/media/{name}")
async def media(name: str) -> FileResponse:
    path = data.media_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="未找到该媒体文件")
    return FileResponse(path, media_type="image/gif")


@app.get("/api/chat/history")
async def chat_history(session_id: str = "", exercise_id: str = "") -> dict[str, Any]:
    """返回该会话的用户可见历史（不含 system 动作上下文），供前端恢复聊天记录。"""
    if not session_id or not exercise_id:
        return {"messages": []}
    return {"messages": chat.history(session_id, exercise_id)}


@app.post("/api/chat/reset")
async def chat_reset(payload: ResetIn) -> dict[str, bool]:
    chat.reset_session(payload.session_id)
    return {"ok": True}


@app.post("/api/chat")
async def chat_stream(payload: ChatIn, request: Request) -> StreamingResponse:
    exercise = data.get_exercise(payload.exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail="未找到该动作")
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="消息不能为空")

    try:
        cfg = chat.load_config()
    except chat.ConfigError as exc:
        reason = str(exc)

        async def config_error() -> AsyncIterator[str]:
            yield chat.sse("error", {"message": reason})
        return _stream(config_error())

    chat.get_session(payload.session_id, payload.exercise_id, exercise)
    # 首轮在此注入动作上下文（system），此后不再重复
    chat.append_user(payload.session_id, message, exercise)
    messages = chat.snapshot(payload.session_id)
    turn = chat.turn_count(payload.session_id)

    async def events() -> AsyncIterator[str]:
        yield chat.sse(
            "meta",
            {"session_id": payload.session_id, "exercise_id": payload.exercise_id, "turn": turn},
        )
        parts: list[str] = []
        finished = False
        try:
            async for text in chat.stream_upstream(cfg, messages):
                if await request.is_disconnected():
                    return
                parts.append(text)
                yield chat.sse("delta", {"text": text})
            full = "".join(parts)
            if not full:
                yield chat.sse("error", {"message": "模型服务暂时不可用，请稍后重试"})
                return
            chat.append_assistant(payload.session_id, full)
            finished = True
            yield chat.sse("done", {"ok": True})
        except asyncio.CancelledError:
            raise
        except Exception:
            yield chat.sse("error", {"message": "模型服务暂时不可用，请稍后重试"})
        finally:
            # 上游失败或客户端中断：回滚本轮 user 消息，避免历史残留无回复提问
            if not finished:
                chat.drop_last_user(payload.session_id, message)

    return _stream(events())


def _stream(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "total": data.TOTAL, "config": chat.CONFIG_FILE.is_file()}


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
