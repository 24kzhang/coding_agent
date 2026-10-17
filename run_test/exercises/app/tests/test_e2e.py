"""Playwright 真实浏览器验收：筛选、弹窗、流式聊天、会话隔离与键盘操作。"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect, sync_playwright

APP = Path(__file__).resolve().parent.parent
PORT = 8125
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = APP / "shots"


def wait_ready(timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/api/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("服务未就绪")


@pytest.fixture(scope="module")
def server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=APP,
    )
    try:
        wait_ready()
        yield BASE
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture()
def page(browser, server):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    pg.goto(server + "/")
    pg.wait_for_selector(".card")
    yield pg
    ctx.close()


def count_text(page: Page) -> str:
    return page.locator("#count").inner_text()


def test_home_cards_and_gif_only(page: Page) -> None:
    cards = page.locator(".card")
    assert cards.count() > 0
    assert "1,324" in count_text(page)
    srcs = page.eval_on_selector_all(".card img", "els => els.map(e => e.getAttribute('src'))")
    assert srcs
    for src in srcs:
        assert src.startswith("/media/") and src.endswith(".gif"), src
    dims = page.eval_on_selector_all(".card img", "els => els.map(e => [e.getAttribute('width'), e.getAttribute('height'), e.getAttribute('alt')])")
    for w, h, alt in dims:
        assert w and h and alt
    page.screenshot(path=str(SHOTS / "home.png"), full_page=False)


def test_no_static_image_requests(page: Page) -> None:
    urls: list[str] = []
    page.on("request", lambda req: urls.append(req.url))
    page.reload()
    page.wait_for_selector(".card")
    page.wait_for_timeout(600)
    bad = [u for u in urls if "images/" in u or ".jpg" in u or ".jpeg" in u or ".png" in u]
    assert bad == []
    assert any("/media/" in u and u.endswith(".gif") for u in urls)


def test_filter_combine_and_clear(page: Page) -> None:
    page.locator('.filter-row[data-facet="category"] .chip[data-value="腰部"]').click()
    page.wait_for_function("document.querySelector('#count').innerText.includes('169')")
    assert "category=" in page.url
    page.locator('.filter-row[data-facet="equipment"] .chip[data-value="自重"]').click()
    page.wait_for_function("!document.querySelector('#grid').getAttribute('aria-busy').includes('true')")
    page.wait_for_timeout(300)
    combined = count_text(page)
    assert "equipment=" in page.url and "category=" in page.url
    assert combined != "共 169 个动作"
    page.locator("#clear").click()
    page.wait_for_function("document.querySelector('#count').innerText.includes('1,324')")
    assert page.url.endswith("/")
    assert page.locator("#clear").is_disabled()


def test_empty_state(page: Page) -> None:
    page.fill("#q", "不存在的动作关键词xyz")
    page.wait_for_selector("#empty:not([hidden])")
    expect(page.locator("#empty")).to_contain_text("没有匹配的动作")
    page.locator("#empty-clear").click()
    page.wait_for_function("document.querySelector('#count').innerText.includes('1,324')")


def test_detail_dialog_all_fields(page: Page, server) -> None:
    first_id = page.locator(".card").first.get_attribute("data-id")
    with urllib.request.urlopen(f"{server}/api/exercises/{first_id}", timeout=10) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    page.locator(".card").first.click()
    dialog = page.locator("#modal")
    expect(dialog).to_be_visible()
    expect(page.locator("#m-title")).to_have_text(raw["name"])
    expect(page.locator("#m-desc")).to_have_text(raw["instructions"]["zh"])
    assert page.locator("#m-steps li").count() == len(raw["instruction_steps"]["zh"])
    facts = page.locator("#m-facts").inner_text()
    assert raw["target"] in facts and raw["equipment"] in facts
    for muscle in raw["secondary_muscles"]:
        assert muscle in facts
    meta = page.locator("#m-meta").inner_text()
    assert raw["media_id"] in meta and raw["attribution"] in meta
    assert raw["gif"].split("/")[-1] in meta
    # 原始 image / gif_url 完整路径以纯文本展示，且不加载静图
    assert raw["image"] in meta
    assert raw["gif_url"] in meta
    gif = page.locator("#m-gif").get_attribute("src")
    assert gif == "/media/" + raw["gif_url"].split("/")[-1]
    assert page.locator("#modal img").count() == 1
    page.screenshot(path=str(SHOTS / "dialog.png"))


def test_chat_stream_two_turns_and_payload(page: Page) -> None:
    payloads: list[dict] = []
    page.on(
        "request",
        lambda req: payloads.append(copy.deepcopy(req.post_data_json))
        if req.url.endswith("/api/chat") and req.method == "POST"
        else None,
    )
    page.locator(".card").first.click()
    expect(page.locator("#modal")).to_be_visible()
    page.fill("#msg", "这个动作主要练哪些肌肉？")
    page.keyboard.press("Enter")

    bot = page.locator(".bubble.bot").last
    page.wait_for_function(
        """() => {
            const bots = document.querySelectorAll('.bubble.bot');
            const last = bots[bots.length - 1];
            return last && !last.classList.contains('thinking') && last.innerText.length > 0;
        }""",
        timeout=120000,
    )
    lengths: list[int] = []
    for _ in range(40):
        lengths.append(len(bot.inner_text()))
        if page.locator("#send").is_enabled():
            break
        page.wait_for_timeout(150)
    page.wait_for_function("document.querySelector('#send').disabled === false", timeout=120000)
    final_len = len(bot.inner_text())
    assert final_len > 0
    assert not bot.locator(".retry").count(), bot.inner_text()
    assert page.locator(".bubble.user").count() == 1
    page.screenshot(path=str(SHOTS / "chat.png"))

    page.fill("#msg", "那新手要注意什么？")
    page.click("#send")
    page.wait_for_function(
        "document.querySelectorAll('.bubble.user').length === 2 && document.querySelector('#send').disabled === false",
        timeout=120000,
    )
    assert len(payloads) == 2
    first, second = payloads
    assert set(first) == {"session_id", "exercise_id", "message"}
    assert first["session_id"] == second["session_id"]
    assert first["exercise_id"] == second["exercise_id"] == "0001"
    assert second["message"] == "那新手要注意什么？"
    assert "这个动作主要练哪些肌肉？" not in json.dumps(second, ensure_ascii=False)
    assert page.locator(".bubble.bot").last.inner_text().strip()


def test_session_isolation_between_exercises(page: Page) -> None:
    first_card = page.locator(".card").first
    first_id = first_card.get_attribute("data-id")
    first_card.click()
    expect(page.locator("#modal")).to_be_visible()
    first_sid = page.evaluate(f"sessionStorage.getItem('chat:{first_id}')")
    assert first_sid
    page.click("#close")
    expect(page.locator("#modal")).to_be_hidden()

    second_card = page.locator(".card").nth(1)
    second_id = second_card.get_attribute("data-id")
    assert second_id != first_id
    second_card.click()
    expect(page.locator("#modal")).to_be_visible()
    second_sid = page.evaluate(f"sessionStorage.getItem('chat:{second_id}')")
    assert second_sid and second_sid != first_sid
    assert page.locator(".bubble.user").count() == 0
    assert page.locator(".bubble.bot").count() == 1


def test_reopen_same_exercise_restores_history(page: Page) -> None:
    """同一动作关闭再打开、刷新后恢复可见历史；不同动作仍隔离。"""
    first = page.locator(".card").first
    first_id = first.get_attribute("data-id")
    first.click()
    expect(page.locator("#modal")).to_be_visible()
    page.fill("#msg", "这个动作主要练哪些肌肉？")
    page.click("#send")
    page.wait_for_function(
        "document.querySelectorAll('.bubble.user').length === 1 && document.querySelector('#send').disabled === false",
        timeout=120000,
    )
    first_answer = page.locator(".bubble.bot").last.inner_text().strip()
    assert first_answer
    page.click("#close")
    expect(page.locator("#modal")).to_be_hidden()

    # 重新打开同一动作：可见历史恢复，且与模型历史一致
    page.locator(f'.card[data-id="{first_id}"]').click()
    expect(page.locator("#modal")).to_be_visible()
    page.wait_for_function("document.querySelectorAll('.bubble.user').length === 1", timeout=30000)
    assert page.locator(".bubble.user").first.inner_text().strip() == "这个动作主要练哪些肌肉？"
    assert page.locator(".bubble.bot").last.inner_text().strip() == first_answer
    assert "system" not in page.locator("#log").inner_text().lower()
    page.click("#close")
    expect(page.locator("#modal")).to_be_hidden()

    # 刷新页面后仍能恢复
    page.reload()
    page.wait_for_selector(".card")
    page.locator(f'.card[data-id="{first_id}"]').click()
    expect(page.locator("#modal")).to_be_visible()
    page.wait_for_function("document.querySelectorAll('.bubble.user').length === 1", timeout=30000)
    assert page.locator(".bubble.bot").last.inner_text().strip() == first_answer
    page.click("#close")
    expect(page.locator("#modal")).to_be_hidden()

    # 另一个动作：历史为空，只有欢迎语
    other = page.locator(f'.card:not([data-id="{first_id}"])').first
    other.click()
    expect(page.locator("#modal")).to_be_visible()
    page.wait_for_timeout(500)
    assert page.locator(".bubble.user").count() == 0
    assert page.locator(".bubble.bot").count() == 1
    page.keyboard.press("Escape")
    expect(page.locator("#modal")).to_be_hidden()


def test_keyboard_esc_tab_and_stop(page: Page) -> None:
    page.locator(".card").first.click()
    expect(page.locator("#modal")).to_be_visible()
    expect(page.locator("#m-id")).to_contain_text("动作编号")
    expect(page.locator("#close")).to_be_focused()
    page.keyboard.press("Tab")
    focused = page.evaluate(
        "(() => { const el = document.activeElement; return el ? (el.id || el.tagName) : ''; })()"
    )
    assert focused and focused != "BODY"
    outline = page.evaluate("getComputedStyle(document.activeElement).outlineStyle")
    assert outline != "none"

    page.fill("#msg", "请详细说明这个动作的完整训练安排和注意事项")
    page.click("#send")
    expect(page.locator("#stop")).to_be_visible()
    page.wait_for_function(
        """() => {
            const bots = document.querySelectorAll('.bubble.bot');
            const last = bots[bots.length - 1];
            return last && !last.classList.contains('thinking') && last.innerText.length > 0;
        }""",
        timeout=120000,
    )
    if page.locator("#stop").is_visible():
        page.click("#stop")
        page.wait_for_function("document.querySelector('#send').disabled === false", timeout=30000)
        expect(page.locator("#stop")).to_be_hidden()
        assert "中断" in page.locator(".bubble.bot").last.inner_text()
    else:
        # 回复在点击前已完成，中断路径改由下一轮验证
        page.fill("#msg", "再详细展开一下动作要点")
        page.click("#send")
        expect(page.locator("#stop")).to_be_visible()
        page.wait_for_function(
            """() => {
                const bots = document.querySelectorAll('.bubble.bot');
                const last = bots[bots.length - 1];
                return last && !last.classList.contains('thinking') && last.innerText.length > 0;
            }""",
            timeout=120000,
        )
        page.click("#stop", force=True)
        page.wait_for_function("document.querySelector('#send').disabled === false", timeout=30000)
        expect(page.locator("#stop")).to_be_hidden()
        assert "中断" in page.locator(".bubble.bot").last.inner_text()

    page.keyboard.press("Escape")
    expect(page.locator("#modal")).to_be_hidden()
    assert page.evaluate("document.body.style.overflow") == ""


# ── 可控网络延迟：历史响应晚到、旧流结束不得污染新面板 ──────────────

# 被挂起的请求：{"route": Route, "url": str}
HELD: list[dict] = []


def hold_route(route) -> None:
    """挂起请求：回调内只登记，不阻塞、不跨线程调用 Playwright。

    同步 Playwright 的 Route 绑定在事件循环线程上，回调里不能阻塞等待，
    也不能跨线程调用 continue_/fulfill；放行由主线程的 release() 完成。
    """
    HELD.append({"route": route, "url": route.request.url})


def release(page: Page, seconds: float = 0.0) -> None:
    """在主线程放行被挂起的请求。

    同步 API 的 Route 只能在 Playwright 事件循环线程上操作，因此这里用
    page.wait_for_timeout 驱动 dispatcher 后再调用 continue_。
    """
    if seconds:
        page.wait_for_timeout(seconds * 1000)
    while HELD:
        item = HELD.pop()
        item["route"].continue_()


def test_route_hold_and_release(page: Page) -> None:
    """自检：被挂起的请求可由主线程放行，且不产生跨线程异常。"""
    HELD.clear()
    page.route("**/api/chat/history*", hold_route)
    try:
        # fire-and-forget：把 fetch Promise 存到 window.__probe 后立即返回，
        # evaluate 不等待响应；否则它会等被挂起的请求而阻塞 release。
        # 同时置 __done=false，供后续以可轮询函数观测完成（不用 async 谓词）。
        page.evaluate(
            "() => { window.__done = false;"
            " window.__probe = fetch('/api/chat/history"
            "?session_id=probe&exercise_id=0001')"
            ".then(r => r.json()).then(() => { window.__done = true; });"
            " return true; }"
        )
        wait_held(page, 1)
        assert "/api/chat/history" in HELD[0]["url"]
        release(page)
        # 放行后 Promise 应真正 resolve 并拿到响应体（证明不是仅解除拦截）
        page.wait_for_function("window.__done === true", timeout=15000)
        assert HELD == []
    finally:
        cleanup(page, "**/api/chat/history*")


def wait_held(page: Page, count: int = 1, timeout: float = 20.0) -> None:
    """等待主线程观察到指定数量的挂起请求。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(HELD) >= count:
            return
        page.wait_for_timeout(50)
    raise AssertionError(f"未观察到挂起的请求，当前 {len(HELD)} 个")


def cleanup(page: Page, pattern: str) -> None:
    """异常安全清理：放行残留路由并解除拦截。"""
    try:
        release(page)
    finally:
        page.unroute(pattern)


def _open_first(page: Page) -> str:
    card = page.locator(".card").first
    first_id = card.get_attribute("data-id")
    card.click()
    expect(page.locator("#modal")).to_be_visible()
    expect(page.locator("#m-title")).not_to_have_text("正在加载…")
    return first_id


def test_late_history_does_not_overwrite_new_message(page: Page) -> None:
    """历史响应晚到时：恢复期间禁止发送；恢复后新消息不被晚到历史清空。"""
    first_id = _open_first(page)
    page.fill("#msg", "这个动作主要练哪些肌肉？")
    page.click("#send")
    page.wait_for_function(
        "document.querySelectorAll('.bubble.user').length === 1 && document.querySelector('#send').disabled === false",
        timeout=120000,
    )
    page.click("#close")
    expect(page.locator("#modal")).to_be_hidden()

    page.route("**/api/chat/history*", hold_route)
    try:
        page.locator(f'.card[data-id="{first_id}"]').click()
        expect(page.locator("#modal")).to_be_visible()
        # 主线程等待路由真正被挂起，再操作页面
        wait_held(page, 1)
        assert "/api/chat/history" in HELD[0]["url"]
        # 历史请求已被挂起：发送入口被恢复状态拦截，输入内容保留并给出提示
        page.fill("#msg", "那新手要注意什么？")
        page.click("#send")
        page.wait_for_timeout(300)
        assert page.locator(".bubble.user").count() == 0
        assert page.input_value("#msg") == "那新手要注意什么？"
        expect(page.locator("#chat-hint")).to_be_visible()
        # 主线程明确放行被挂起的历史响应
        release(page)
        page.wait_for_function("document.querySelectorAll('.bubble.user').length === 1", timeout=30000)
        assert page.locator(".bubble.user").first.inner_text().strip() == "这个动作主要练哪些肌肉？"
        # 恢复完成后发送新消息，晚到的历史不得再清空列表
        page.fill("#msg", "那新手要注意什么？")
        page.click("#send")
        page.wait_for_function(
            "document.querySelectorAll('.bubble.user').length === 2 && document.querySelector('#send').disabled === false",
            timeout=120000,
        )
        page.wait_for_timeout(1800)
        users = page.locator(".bubble.user")
        assert users.count() == 2
        assert users.nth(0).inner_text().strip() == "这个动作主要练哪些肌肉？"
        assert users.nth(1).inner_text().strip() == "那新手要注意什么？"
    finally:
        cleanup(page, "**/api/chat/history*")
    page.keyboard.press("Escape")
    expect(page.locator("#modal")).to_be_hidden()


def test_old_stream_end_does_not_touch_new_panel(page: Page) -> None:
    """旧动作的流在新动作面板打开后才结束：不得清除新流控制器或改写新面板。"""
    HELD.clear()
    page.route("**/api/chat", hold_route)
    first = page.locator(".card").first
    first_id = first.get_attribute("data-id")
    first.click()
    expect(page.locator("#modal")).to_be_visible()
    page.fill("#msg", "这个动作主要练哪些肌肉？")
    page.click("#send")
    expect(page.locator("#stop")).to_be_visible()
    # 主线程确认旧请求确实被挂起（尚未结束），而不是靠固定等待猜测时序
    wait_held(page, 1)
    assert "/api/chat" in HELD[0]["url"]

    page.click("#close")
    expect(page.locator("#modal")).to_be_hidden()

    second = page.locator(f'.card:not([data-id="{first_id}"])').first
    second_id = second.get_attribute("data-id")
    second.click()
    expect(page.locator("#modal")).to_be_visible()
    expect(page.locator("#m-title")).not_to_have_text("正在加载…")
    # 新面板已打开，此时才放行旧流：其结束事件真实发生在切换之后
    release(page)
    # 观察旧流的结束对当前页面产生的实际效果：停止按钮消失、发送可用
    page.wait_for_function("document.querySelector('#stop').hidden === true", timeout=30000)
    assert page.locator(".bubble.user").count() == 0
    assert page.locator(".bubble.bot").count() == 1
    assert page.locator("#send").is_enabled()
    assert page.locator("#msg").is_enabled()
    assert page.evaluate(f"sessionStorage.getItem('chat:{second_id}')") != page.evaluate(
        f"sessionStorage.getItem('chat:{first_id}')"
    )
    # 新面板仍可正常发起新流；解除拦截让新请求直达真实服务
    page.unroute("**/api/chat")
    page.fill("#msg", "这个动作适合新手吗？")
    page.click("#send")
    expect(page.locator("#stop")).to_be_visible()
    page.wait_for_function(
        "document.querySelectorAll('.bubble.user').length === 1 && document.querySelector('#send').disabled === false",
        timeout=120000,
    )
    assert page.locator(".bubble.user").first.inner_text().strip() == "这个动作适合新手吗？"
    assert page.locator(".bubble.bot").last.inner_text().strip()
    assert not page.locator(".bubble.bot").last.locator(".retry").count()
    page.keyboard.press("Escape")
    expect(page.locator("#modal")).to_be_hidden()


def test_detail_failure_clears_previous_info(page: Page) -> None:
    """详情加载失败时不得残留上一个动作的信息。"""
    first = page.locator(".card").first
    first.click()
    expect(page.locator("#modal")).to_be_visible()
    expect(page.locator("#m-title")).not_to_have_text("正在加载…")
    old_title = page.locator("#m-title").inner_text()
    page.keyboard.press("Escape")
    expect(page.locator("#modal")).to_be_hidden()

    page.route("**/api/exercises/*", lambda route: route.fulfill(status=500, body="{}"))
    page.locator(".card").nth(1).click()
    expect(page.locator("#modal")).to_be_visible()
    expect(page.locator("#m-title")).to_have_text("动作详情加载失败")
    assert page.locator("#m-title").inner_text() != old_title
    assert page.locator("#m-desc").inner_text() == "请关闭弹窗后重试。"
    assert page.locator("#m-steps li").count() == 0
    assert page.locator("#m-facts").inner_text().strip() == ""
    assert page.locator("#m-meta").inner_text().strip() == ""
    assert page.locator("#m-gif").get_attribute("src") is None
    page.unroute("**/api/exercises/*")
    page.keyboard.press("Escape")
    expect(page.locator("#modal")).to_be_hidden()
