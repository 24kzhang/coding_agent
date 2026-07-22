"""健身动作网站后端服务"""
import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

app = FastAPI(title="健身动作网站")


class ChatRequest(BaseModel):
    """对话请求模型"""
    message: str = Field(..., min_length=1, description="用户消息")
    history: list[dict] = Field(default_factory=list, description="对话历史")
    is_first: bool = Field(default=False, description="是否首次对话")


# 数据文件路径（基于 __file__ 的绝对路径）
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "exercises-dataset" / "data" / "exercises_zh.json"
MEDIA_DIR = BASE_DIR / "exercises-dataset"


def load_exercises() -> list[dict]:
    """加载动作数据"""
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


EXERCISES = load_exercises()
EXERCISE_MAP = {ex["id"]: ex for ex in EXERCISES}


def extract_options() -> dict:
    """提取所有筛选选项"""
    categories = sorted(set(ex["category"] for ex in EXERCISES if ex.get("category")))
    muscles = sorted(set(ex["target"] for ex in EXERCISES if ex.get("target")))
    equipment = sorted(set(ex["equipment"] for ex in EXERCISES if ex.get("equipment")))
    return {"categories": categories, "muscles": muscles, "equipment": equipment}


OPTIONS = extract_options()
PAGE_SIZE = 24

# 静态文件和模板目录
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# 挂载静态文件和模板
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 固定的助手系统提示
SYSTEM_PROMPT = "你是一个专业的健身助手，擅长解答健身动作相关问题，提供训练建议和注意事项。回答要简洁专业，使用中文。"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页"""
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/options")
async def get_options():
    """返回所有筛选选项"""
    return OPTIONS


@app.get("/api/exercises")
async def get_exercises(
    page: int = Query(1, ge=1),
    category: str | None = Query(None),
    muscle: str | None = Query(None),
    equipment: str | None = Query(None),
):
    """分页+筛选返回动作列表"""
    filtered = EXERCISES
    if category:
        filtered = [ex for ex in filtered if ex.get("category") == category]
    if muscle:
        filtered = [ex for ex in filtered if ex.get("target") == muscle]
    if equipment:
        filtered = [ex for ex in filtered if ex.get("equipment") == equipment]

    total = len(filtered)
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    items = filtered[start:end]

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": PAGE_SIZE,
        "total_pages": (total + PAGE_SIZE - 1) // PAGE_SIZE if total > 0 else 1,
    }


@app.get("/api/exercises/{exercise_id}")
async def get_exercise(exercise_id: str):
    """返回单个动作完整数据"""
    ex = EXERCISE_MAP.get(exercise_id)
    if not ex:
        raise HTTPException(status_code=404, detail="动作不存在")
    return ex


@app.post("/api/chat/{exercise_id}")
async def chat(exercise_id: str, req: ChatRequest):
    """SSE流式对话接口"""
    ex = EXERCISE_MAP.get(exercise_id)
    if not ex:
        raise HTTPException(status_code=404, detail="动作不存在")

    api_key = os.environ.get("LONGCAT_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="未配置 LONGCAT_API_KEY 环境变量")

    # 构建消息列表：固定系统提示 + 仅首次的完整动作 JSON + history + 当前用户消息
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if req.is_first:
        exercise_info = (
            f"以下是用户正在查看的健身动作的完整信息：\n"
            f"{json.dumps(ex, ensure_ascii=False, indent=2)}\n\n"
            "请根据这个动作的信息回答用户的问题。"
        )
        messages.append({"role": "system", "content": exercise_info})

    # 添加历史对话
    for msg in req.history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 添加当前用户消息
    messages.append({"role": "user", "content": req.message})

    async def event_stream():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "https://api.longcat.chat/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "LongCat-2.0",
                    "messages": messages,
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"data: {json.dumps({'error': f'API请求失败: {response.status_code} {error_text.decode()}'}, ensure_ascii=False)}\n\n"
                    return
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if not isinstance(choices, list) or len(choices) == 0:
                                continue
                            first = choices[0]
                            if not isinstance(first, dict):
                                continue
                            delta = first.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                        except json.JSONDecodeError:
                            continue

    return StreamingResponse(event_stream(), media_type="text/event-stream")
