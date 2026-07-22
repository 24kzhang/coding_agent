"""健身动作网站核心功能测试"""
import json
import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from main import EXERCISE_MAP, app, load_exercises

# 确保测试时不依赖真实 API key
os.environ.setdefault("LONGCAT_API_KEY", "test-key-for-testing")


class MockResponse:
    """模拟 httpx 流式响应对象"""
    status_code = 200

    async def aiter_lines(self):
        """返回标准 SSE 格式的行，包含空 choices 控制块"""
        yield 'data: {"choices": [{"delta": {"content": "你好"}, "finish_reason": null}]}'
        yield 'data: {"choices": []}'
        yield 'data: {"choices": [{"delta": {"content": "！"}, "finish_reason": null}]}'
        yield 'data: [DONE]'

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class MockStream:
    """模拟 stream() 返回的异步上下文对象"""
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class MockAsyncClient:
    """模拟 httpx.AsyncClient"""
    stream_calls = []

    def __init__(self, *args, **kwargs):
        pass

    def stream(self, *args, **kwargs):
        MockAsyncClient.stream_calls.append(kwargs.get("json"))
        return MockStream(MockResponse())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture(scope="session")
def exercises_data():
    """加载测试用动作数据"""
    return load_exercises()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """创建异步测试客户端"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── 数据接口测试 ──────────────────────────────────────

@pytest.mark.anyio
async def test_get_exercises_default_page(client):
    """测试默认分页：第一页返回24条"""
    resp = await client.get("/api/exercises")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert data["page"] == 1
    assert data["page_size"] == 24
    assert len(data["items"]) == 24
    assert data["total"] == 1324


@pytest.mark.anyio
async def test_get_exercises_page2(client):
    """测试第二页分页正确性"""
    resp = await client.get("/api/exercises?page=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 2
    assert len(data["items"]) == 24


@pytest.mark.anyio
async def test_get_exercises_last_page(client):
    """测试最后一页数据量正确"""
    resp = await client.get("/api/exercises?page=56")
    assert resp.status_code == 200
    data = resp.json()
    # 1324 / 24 = 55.17，第56页应有 1324 - 55*24 = 4 条
    assert data["page"] == 56
    assert len(data["items"]) == 1324 - 55 * 24


@pytest.mark.anyio
async def test_get_exercises_filter_category(client):
    """测试按分类筛选"""
    resp = await client.get("/api/exercises?category=腰部")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    for item in data["items"]:
        assert item["category"] == "腰部"


@pytest.mark.anyio
async def test_get_exercises_filter_muscle(client):
    """测试按目标肌肉筛选"""
    resp = await client.get("/api/exercises?muscle=腹肌")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    for item in data["items"]:
        assert item["target"] == "腹肌"


@pytest.mark.anyio
async def test_get_exercises_filter_equipment(client):
    """测试按器材筛选"""
    resp = await client.get("/api/exercises?equipment=自重")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    for item in data["items"]:
        assert item["equipment"] == "自重"


@pytest.mark.anyio
async def test_get_exercises_combined_filter(client):
    """测试组合筛选"""
    resp = await client.get("/api/exercises?category=腰部&equipment=自重")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    for item in data["items"]:
        assert item["category"] == "腰部"
        assert item["equipment"] == "自重"


@pytest.mark.anyio
async def test_get_exercises_no_results(client):
    """测试无匹配结果"""
    resp = await client.get("/api/exercises?category=不存在的分类")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert len(data["items"]) == 0


# ── 详情接口测试 ──────────────────────────────────────

@pytest.mark.anyio
async def test_get_exercise_detail(client, exercises_data):
    """测试获取单个动作完整数据"""
    first_id = exercises_data[0]["id"]
    resp = await client.get(f"/api/exercises/{first_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == first_id
    assert "name" in data
    assert "category" in data
    assert "target" in data
    assert "equipment" in data
    assert "gif_url" in data
    assert "instructions" in data


@pytest.mark.anyio
async def test_get_exercise_not_found(client):
    """测试动作不存在时返回404"""
    resp = await client.get("/api/exercises/9999")
    assert resp.status_code == 404


# ── 选项接口测试 ──────────────────────────────────────

@pytest.mark.anyio
async def test_get_options(client):
    """测试获取筛选选项"""
    resp = await client.get("/api/options")
    assert resp.status_code == 200
    data = resp.json()
    assert "categories" in data
    assert "muscles" in data
    assert "equipment" in data
    assert len(data["categories"]) > 0
    assert len(data["muscles"]) > 0
    assert len(data["equipment"]) > 0


# ── 对话接口测试 ──────────────────────────────────────

@pytest.mark.anyio
async def test_chat_sse_response(client):
    """测试对话接口返回SSE流式响应"""
    with patch("main.httpx.AsyncClient", MockAsyncClient):
        resp = await client.post(
            "/api/chat/0001",
            json={"message": "这个动作怎么做？", "history": []}
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "你好" in body
        assert "！" in body


@pytest.mark.anyio
async def test_chat_first_request_includes_exercise_json(client):
    """测试首次请求 is_first=true，messages 包含完整动作 JSON"""
    MockAsyncClient.stream_calls = []
    with patch("main.httpx.AsyncClient", MockAsyncClient):
        resp = await client.post(
            "/api/chat/0001",
            json={"message": "这个动作怎么做？", "history": [], "is_first": True}
        )
        assert resp.status_code == 200
        assert len(MockAsyncClient.stream_calls) == 1
        sent_json = MockAsyncClient.stream_calls[0]
        assert sent_json is not None
        messages = sent_json.get("messages", [])
        exercise_json = json.dumps(EXERCISE_MAP["0001"], ensure_ascii=False, indent=2)
        found = any(exercise_json in msg.get("content", "") for msg in messages)
        assert found, "首次请求的 messages 应包含完整动作 JSON"


@pytest.mark.anyio
async def test_chat_subsequent_request_no_exercise_json(client):
    """测试后续请求 is_first=false，历史保留，当前消息在最后，不再包含完整动作 JSON"""
    history = [
        {"role": "user", "content": "这个动作怎么做？"},
        {"role": "assistant", "content": "回答内容"}
    ]
    MockAsyncClient.stream_calls = []
    with patch("main.httpx.AsyncClient", MockAsyncClient):
        resp = await client.post(
            "/api/chat/0001",
            json={"message": "另一个问题", "history": history, "is_first": False}
        )
        assert resp.status_code == 200
        assert len(MockAsyncClient.stream_calls) == 1
        sent_json = MockAsyncClient.stream_calls[0]
        messages = sent_json.get("messages", [])
        # 验证历史 user/assistant 消息保留
        assert {"role": "user", "content": "这个动作怎么做？"} in messages
        assert {"role": "assistant", "content": "回答内容"} in messages
        # 验证当前用户消息在最后
        assert messages[-1] == {"role": "user", "content": "另一个问题"}
        # 验证不再包含完整动作 JSON
        exercise_json = json.dumps(EXERCISE_MAP["0001"], ensure_ascii=False, indent=2)
        found = any(exercise_json in msg.get("content", "") for msg in messages)
        assert not found, "后续请求的 messages 不应包含完整动作 JSON"


@pytest.mark.anyio
async def test_chat_exercise_not_found(client):
    """测试对话时动作不存在返回404"""
    resp = await client.post(
        "/api/chat/9999",
        json={"message": "你好", "history": []}
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_chat_missing_message(client):
    """测试缺少消息字段"""
    resp = await client.post(
        "/api/chat/0001",
        json={"history": []}
    )
    assert resp.status_code == 422


# ── 首页测试 ──────────────────────────────────────────

@pytest.mark.anyio
async def test_index_page(client):
    """测试首页可正常访问"""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


# ── 媒体路径测试 ──────────────────────────────────────

def test_gif_url_format(exercises_data):
    """测试GIF链接格式正确"""
    for ex in exercises_data[:10]:
        assert ex["gif_url"].startswith("videos/")
        assert ex["gif_url"].endswith(".gif")


def test_image_url_format(exercises_data):
    """测试图片链接格式正确"""
    for ex in exercises_data[:10]:
        assert ex["image"].startswith("images/")


@pytest.mark.anyio
async def test_chat_empty_choices_does_not_crash(client):
    """测试空 choices 控制块不会导致 SSE 流崩溃"""
    with patch("main.httpx.AsyncClient", MockAsyncClient):
        resp = await client.post(
            "/api/chat/0001",
            json={"message": "这个动作怎么做？", "history": []}
        )
        assert resp.status_code == 200
        body = resp.text
        assert "你好" in body
        assert "！" in body
        assert "[DONE]" in body
# ── DOM 合同回归测试 ──────────────────────────────────────

# 前端 app.js 静态引用的 DOM ID，必须与 templates/index.html 保持一致
# 注意：typing-indicator 由 app.js 在运行时动态创建（addTypingIndicator），
# 不属于静态模板，因此单独放入 APP_JS_DYNAMIC_IDS。
APP_JS_REQUIRED_IDS = [
    "filter-category", "filter-muscle", "filter-equipment",
    "btn-reset", "btn-prev", "btn-next",
    "exercise-grid", "total-count", "page-info",
    "modal-overlay", "modal-close",
    "detail-gif", "detail-name", "detail-category",
    "detail-body-part", "detail-target", "detail-muscle-group",
    "detail-secondary", "detail-equipment",
    "detail-instructions", "detail-steps",
    "chat-messages", "chat-input", "chat-send",
]

# 前端 app.js 运行时动态创建的 DOM ID（不应要求静态存在于 index.html）
APP_JS_DYNAMIC_IDS = [
    "typing-indicator",
]

# 前端 app.js 动态创建的 CSS 类，必须与 static/style.css 保持一致
APP_JS_REQUIRED_CLASSES = [
    "chat-msg", "user", "assistant",
    "typing-indicator", "typing-dot",
]


def _read_relative(relative_path: str) -> str:
    """读取项目根目录下的文件内容"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, relative_path), encoding="utf-8") as f:
        return f.read()


@pytest.mark.anyio
async def test_index_contains_all_app_js_element_ids(client):
    """回归测试：index.html 必须包含 app.js 引用的所有 DOM ID"""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    missing = [dom_id for dom_id in APP_JS_REQUIRED_IDS if f'id="{dom_id}"' not in html]
    assert not missing, f"index.html 缺少以下 DOM ID: {missing}"


def test_style_css_contains_app_js_dynamic_classes():
    """回归测试：style.css 必须定义 app.js 动态创建的 CSS 类"""
    css = _read_relative("static/style.css")
    missing = [cls for cls in APP_JS_REQUIRED_CLASSES if f".{cls}" not in css]
    assert not missing, f"style.css 缺少以下 CSS 类定义: {missing}"


def test_index_contains_required_css_classes():
    """回归测试：index.html 使用的关键 CSS 类必须在 style.css 中定义"""
    html = _read_relative("templates/index.html")
    css = _read_relative("static/style.css")
    # 从 HTML 中提取所有 class 属性值
    import re
    classes_in_html = set()
    for match in re.findall(r'class="([^"]+)"', html):
        for cls in match.split():
            classes_in_html.add(cls)
    # 忽略 JS 动态添加的类（如 hidden）
    js_dynamic = {"hidden"}
    classes_in_html -= js_dynamic
    missing = [cls for cls in classes_in_html if f".{cls}" not in css]
    assert not missing, f"style.css 缺少以下 HTML 中使用的 CSS 类: {missing}"


@pytest.mark.anyio
async def test_favicon_no_404(client):
    """回归测试：首页不应包含会导致 404 的 favicon 链接"""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    # 不应有指向 /favicon.ico 的链接（会导致 404）
    assert 'href="/favicon.ico"' not in html, "不应有指向 /favicon.ico 的链接"
    # 应使用 data URI 内联 favicon
    assert "data:image/svg+xml" in html, "应使用 data URI 内联 favicon"
