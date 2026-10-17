/* 前端交互：筛选、弹窗、聊天流式渲染与键盘操作。 */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const grid = $("#grid");
  const countEl = $("#count");
  const clearBtn = $("#clear");
  const emptyEl = $("#empty");
  const noticeEl = $("#notice");
  const qInput = $("#q");
  const modal = $("#modal");
  const logEl = $("#log");
  const suggestEl = $("#suggest");
  const msgEl = $("#msg");
  const sendBtn = $("#send");
  const stopBtn = $("#stop");

  const nf = new Intl.NumberFormat("zh-CN");
  const state = { category: "", target: "", equipment: "", q: "" };
  let lastCard = null;
  let current = null; // {id, name, sessionId}
  let controller = null;
  let busy = false;
  /* 正在恢复历史的面板令牌；null 表示没有进行中的恢复 */
  let restoringToken = null;

  /* ── URL 状态 ─────────────────────────── */
  function readUrl() {
    const p = new URLSearchParams(location.search);
    state.category = p.get("category") || "";
    state.target = p.get("target") || "";
    state.equipment = p.get("equipment") || "";
    state.q = p.get("q") || "";
    qInput.value = state.q;
    document.querySelectorAll(".filter-row").forEach((row) => {
      const key = row.dataset.facet;
      row.querySelectorAll(".chip").forEach((chip) => {
        const on = chip.dataset.value === state[key];
        chip.classList.toggle("is-on", on);
        chip.setAttribute("aria-pressed", String(on));
      });
    });
  }

  function writeUrl() {
    const p = new URLSearchParams();
    ["category", "target", "equipment", "q"].forEach((k) => {
      if (state[k]) p.set(k, state[k]);
    });
    const qs = p.toString();
    history.replaceState(null, "", qs ? `?${qs}` : location.pathname);
  }

  function hasFilter() {
    return Boolean(state.category || state.target || state.equipment || state.q);
  }

  /* ── 列表渲染 ─────────────────────────── */
  function skeletons(n = 8) {
    grid.innerHTML = "";
    for (let i = 0; i < n; i += 1) {
      const s = document.createElement("div");
      s.className = "skeleton";
      s.setAttribute("aria-hidden", "true");
      grid.appendChild(s);
    }
  }

  function tag(text, key) {
    const span = document.createElement("span");
    span.className = key ? "tag is-key" : "tag";
    span.textContent = text || "暂无数据";
    return span;
  }

  function card(item, index) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "card";
    btn.dataset.id = item.id;
    btn.style.animationDelay = `${Math.min(index, 12) * 22}ms`;
    btn.setAttribute("aria-label", `查看动作：${item.name}`);

    const img = document.createElement("img");
    img.src = `/media/${item.gif}`;
    img.alt = `${item.name} 动作演示`;
    img.width = 180;
    img.height = 180;
    img.loading = index < 8 ? "eager" : "lazy";
    img.decoding = "async";

    const name = document.createElement("span");
    name.className = "card-name";
    name.textContent = item.name;

    const tags = document.createElement("span");
    tags.className = "card-tags";
    tags.append(tag(item.category), tag(item.target), tag(item.equipment));

    btn.append(img, name, tags);
    btn.addEventListener("click", () => openDetail(item.id, btn));
    return btn;
  }

  function showNotice(text, retry) {
    noticeEl.innerHTML = "";
    const span = document.createElement("span");
    span.textContent = text;
    noticeEl.appendChild(span);
    if (retry) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ghost";
      b.textContent = "重试";
      b.addEventListener("click", () => load());
      noticeEl.appendChild(b);
    }
    noticeEl.hidden = false;
  }

  async function load() {
    grid.setAttribute("aria-busy", "true");
    noticeEl.hidden = true;
    emptyEl.hidden = true;
    skeletons();
    const p = new URLSearchParams();
    ["category", "target", "equipment", "q"].forEach((k) => {
      if (state[k]) p.set(k, state[k]);
    });
    try {
      const res = await fetch(`/api/exercises?${p.toString()}`);
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      grid.innerHTML = "";
      const frag = document.createDocumentFragment();
      data.items.forEach((item, i) => frag.appendChild(card(item, i)));
      grid.appendChild(frag);
      countEl.innerHTML = `共 <strong>${nf.format(data.count)}</strong> 个动作`;
      emptyEl.hidden = data.count !== 0;
      clearBtn.disabled = !hasFilter();
    } catch (err) {
      grid.innerHTML = "";
      countEl.textContent = "加载失败";
      showNotice("动作列表加载失败，请检查网络后重试。", true);
    } finally {
      grid.setAttribute("aria-busy", "false");
    }
  }

  /* ── 筛选交互 ─────────────────────────── */
  document.querySelectorAll(".filter-row").forEach((row) => {
    row.addEventListener("click", (e) => {
      const chip = e.target.closest(".chip");
      if (!chip) return;
      state[row.dataset.facet] = chip.dataset.value;
      row.querySelectorAll(".chip").forEach((c) => {
        const on = c === chip;
        c.classList.toggle("is-on", on);
        c.setAttribute("aria-pressed", String(on));
      });
      writeUrl();
      load();
    });
  });

  let timer = 0;
  qInput.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      state.q = qInput.value.trim();
      writeUrl();
      load();
    }, 260);
  });

  function clearAll() {
    state.category = "";
    state.target = "";
    state.equipment = "";
    state.q = "";
    qInput.value = "";
    document.querySelectorAll(".filter-row").forEach((row) => {
      row.querySelectorAll(".chip").forEach((c) => {
        const on = c.dataset.value === "";
        c.classList.toggle("is-on", on);
        c.setAttribute("aria-pressed", String(on));
      });
    });
    writeUrl();
    load();
  }
  clearBtn.addEventListener("click", clearAll);
  $("#empty-clear").addEventListener("click", clearAll);
  window.addEventListener("popstate", () => {
    readUrl();
    load();
  });

  /* ── 详情弹窗 ─────────────────────────── */
  function fact(dl, label, value) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value || "暂无数据";
    dl.append(dt, dd);
  }

  function fmtDate(iso) {
    if (!iso) return "暂无数据";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return new Intl.DateTimeFormat("zh-CN", { dateStyle: "long", timeStyle: "short" }).format(d);
  }

  function renderDetail(item) {
    $("#m-id").textContent = `动作编号 ${item.id}`;
    $("#m-title").textContent = item.name || "暂无数据";
    const tags = $("#m-tags");
    tags.innerHTML = "";
    [item.category, item.body_part, item.target, item.equipment].forEach((v) => tags.appendChild(tag(v, true)));

    const img = $("#m-gif");
    img.src = `/media/${item.gif}`;
    img.alt = `${item.name} 动作演示`;

    $("#m-desc").textContent = (item.instructions && item.instructions.zh) || "暂无数据";
    const steps = $("#m-steps");
    steps.innerHTML = "";
    const list = (item.instruction_steps && item.instruction_steps.zh) || [];
    if (list.length === 0) {
      const li = document.createElement("li");
      li.textContent = "暂无数据";
      steps.appendChild(li);
    } else {
      list.forEach((s) => {
        const li = document.createElement("li");
        li.textContent = s;
        steps.appendChild(li);
      });
    }

    const facts = $("#m-facts");
    facts.innerHTML = "";
    fact(facts, "目标肌肉", item.target);
    fact(facts, "肌肉群", item.muscle_group);
    fact(facts, "协同肌肉", (item.secondary_muscles || []).join("、"));
    fact(facts, "所需器械", item.equipment);
    fact(facts, "动作分类", item.category);
    fact(facts, "身体部位", item.body_part);

    const meta = $("#m-meta");
    meta.innerHTML = "";
    fact(meta, "媒体编号", item.media_id);
    fact(meta, "创建时间", fmtDate(item.created_at));
    fact(meta, "版权说明", item.attribution);
    /* 原始字段完整路径以纯文本展示，不加载静图 */
    fact(meta, "静图路径（image）", item.image);
    fact(meta, "动图路径（gif_url）", item.gif_url);
    fact(meta, "GIF 文件名", item.gif);
  }

  function sessionKey(id) {
    return `chat:${id}`;
  }

  function sessionId(id) {
    let sid = sessionStorage.getItem(sessionKey(id));
    if (!sid) {
      sid = crypto.randomUUID();
      sessionStorage.setItem(sessionKey(id), sid);
    }
    return sid;
  }

  function bubble(role, text) {
    const div = document.createElement("div");
    div.className = `bubble ${role}`;
    div.textContent = text;
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
    return div;
  }

  function welcome() {
    logEl.innerHTML = "";
    bubble("bot", "你好！我是你的动作教练。关于这个动作的要点、常见错误、适合人群，都可以问我。");
    suggestEl.hidden = false;
  }

  /* 从服务端恢复该动作会话的用户可见历史（不含 system 上下文）。
     恢复期间禁止发送；若用户已产生新气泡则不再清空列表，避免抹掉正在流式输出的内容。 */
  async function restoreHistory(sid, id, token) {
    /* 恢复状态按面板令牌隔离：只有当前面板的恢复请求才能改变 restoringToken */
    restoringToken = token;
    try {
      const p = new URLSearchParams({ session_id: sid, exercise_id: id });
      const res = await fetch(`/api/chat/history?${p.toString()}`);
      if (!res.ok) return;
      const data = await res.json();
      if (token !== current || !modal.open) return;
      if (!Array.isArray(data.messages) || data.messages.length === 0) return;
      /* 历史只插入到欢迎语之后，绝不整体清空列表，避免抹掉已产生的新消息 */
      const anchor = logEl.querySelector(".bubble.bot");
      const frag = document.createDocumentFragment();
      data.messages.forEach((m) => {
        const div = document.createElement("div");
        div.className = `bubble ${m.role === "user" ? "user" : "bot"}`;
        div.textContent = m.content;
        frag.appendChild(div);
      });
      if (anchor) anchor.after(frag);
      else logEl.appendChild(frag);
      logEl.scrollTop = logEl.scrollHeight;
      suggestEl.hidden = true;
    } catch (err) {
      /* 历史恢复失败时保留欢迎语，不影响继续提问 */
    } finally {
      /* 只复位自己登记的恢复状态，过期面板不得影响新面板 */
      if (restoringToken === token) restoringToken = null;
    }
  }

  /* 清空上一个动作的详情内容，避免加载失败时残留旧信息 */
  function resetDetail() {
    $("#m-id").textContent = "动作编号 —";
    $("#m-title").textContent = "正在加载…";
    $("#m-tags").innerHTML = "";
    const img = $("#m-gif");
    img.removeAttribute("src");
    img.alt = "";
    $("#m-desc").textContent = "正在加载…";
    $("#m-steps").innerHTML = "";
    $("#m-facts").innerHTML = "";
    $("#m-meta").innerHTML = "";
  }

  async function openDetail(id, trigger) {
    lastCard = trigger || null;
    abortStream();
    welcome();
    msgEl.value = "";
    resetDetail();
    const sid = sessionId(id);
    current = { id, name: "", sessionId: sid };
    const token = current;
    modal.showModal();
    document.body.style.overflow = "hidden";
    $("#close").focus();
    try {
      const res = await fetch(`/api/exercises/${encodeURIComponent(id)}`);
      if (!res.ok) throw new Error(String(res.status));
      const item = await res.json();
      if (token !== current) return;
      current.name = item.name;
      renderDetail(item);
    } catch (err) {
      if (token !== current) return;
      resetDetail();
      $("#m-title").textContent = "动作详情加载失败";
      $("#m-desc").textContent = "请关闭弹窗后重试。";
    }
    restoreHistory(sid, id, token);
  }

  function closeModal() {
    abortStream();
    /* 使当前面板令牌失效：旧流的回调与 finally 不再操作任何面板状态 */
    current = null;
    if (restoringToken) restoringToken = null;
    modal.close();
  }

  modal.addEventListener("close", () => {
    document.body.style.overflow = "";
    if (lastCard) lastCard.focus();
  });
  $("#close").addEventListener("click", closeModal);
  modal.addEventListener("cancel", (e) => {
    e.preventDefault();
    closeModal();
  });

  /* ── 聊天 ─────────────────────────────── */
  function setBusy(on) {
    busy = on;
    sendBtn.disabled = on;
    stopBtn.hidden = !on;
    msgEl.disabled = on;
  }

  function abortStream() {
    if (controller) {
      controller.abort();
      controller = null;
    }
    setBusy(false);
  }

  /* 按空行分帧，同时兼容 \n\n 与 \r\n\r\n 两种换行 */
  function parseFrames(buffer, onEvent) {
    const sep = /\r?\n\r?\n/;
    let match = sep.exec(buffer);
    while (match) {
      const frame = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      let event = "message";
      const dataLines = [];
      frame.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      });
      if (dataLines.length) {
        try {
          onEvent(event, JSON.parse(dataLines.join("\n")));
        } catch (err) {
          /* 忽略无法解析的帧 */
        }
      }
      match = sep.exec(buffer);
    }
    return buffer;
  }

  async function send(text) {
    const message = (text || "").trim();
    /* 历史恢复未完成时禁止发送，避免晚到的历史覆盖新消息 */
    if (!message || busy || !current) return;
    if (restoringToken === current) return;
    const token = current;
    const sid = current.sessionId;
    const exId = current.id;
    suggestEl.hidden = true;
    bubble("user", message);
    const bot = bubble("bot", "正在思考…");
    bot.classList.add("thinking");
    const ctrl = new AbortController();
    controller = ctrl;
    setBusy(true);
    let acc = "";
    let failed = false;

    /* 只有本次请求仍是当前流、且面板仍属于同一动作时才允许写 DOM */
    const alive = () => token === current && controller === ctrl && modal.open;

    const handle = (event, data) => {
      if (!alive()) return;
      if (event === "delta") {
        if (bot.classList.contains("thinking")) {
          bot.classList.remove("thinking");
          bot.textContent = "";
          bot.classList.add("cursor");
        }
        acc += data.text || "";
        bot.textContent = acc;
        logEl.scrollTop = logEl.scrollHeight;
      } else if (event === "error") {
        failed = true;
        bot.classList.remove("thinking", "cursor");
        bot.classList.add("error");
        bot.textContent = data.message || "模型服务暂时不可用，请稍后重试";
        const retry = document.createElement("button");
        retry.type = "button";
        retry.className = "ghost retry";
        retry.textContent = "重试";
        retry.addEventListener("click", () => {
          bot.remove();
          send(message);
        });
        bot.appendChild(retry);
      } else if (event === "done") {
        bot.classList.remove("cursor");
      }
    };

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sid, exercise_id: exId, message }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(String(res.status));
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = parseFrames(buffer, handle);
      }
      if (alive() && !acc && !failed) {
        bot.classList.remove("thinking");
        bot.classList.add("error");
        bot.textContent = "模型服务暂时不可用，请稍后重试";
      }
    } catch (err) {
      if (!alive()) return;
      bot.classList.remove("thinking", "cursor");
      if (err && err.name === "AbortError") {
        bot.classList.add("error");
        bot.textContent = acc ? `${acc}\n（已中断）` : "已中断本次回复。";
      } else {
        bot.classList.add("error");
        bot.textContent = "网络异常，回复失败。请重试。";
      }
    } finally {
      /* 旧流结束不得清除新流的 controller，也不得修改新面板状态 */
      if (controller === ctrl) {
        controller = null;
        setBusy(false);
        if (token === current && modal.open) msgEl.focus();
      }
    }
  }

  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = msgEl.value;
    /* 恢复历史期间不发送，也不清空输入框，避免静默丢弃用户输入 */
    if (restoringToken === current) {
      showChatHint("正在恢复聊天记录，请稍候…");
      return;
    }
    msgEl.value = "";
    send(text);
  });

  function showChatHint(text) {
    let hint = $("#chat-hint");
    if (!hint) {
      hint = document.createElement("p");
      hint.id = "chat-hint";
      hint.className = "chat-hint";
      hint.setAttribute("role", "status");
      hint.setAttribute("aria-live", "polite");
      $("#composer").before(hint);
    }
    hint.textContent = text;
    hint.hidden = false;
    clearTimeout(showChatHint.timer);
    showChatHint.timer = setTimeout(() => {
      hint.hidden = true;
    }, 2600);
  }

  msgEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("#composer").requestSubmit();
    }
  });

  stopBtn.addEventListener("click", () => {
    if (controller) controller.abort();
  });

  suggestEl.addEventListener("click", (e) => {
    const btn = e.target.closest(".suggest-item");
    if (!btn) return;
    send(btn.dataset.q);
  });

  /* ── 弹窗内焦点循环兜底 ───────────────── */
  modal.addEventListener("keydown", (e) => {
    if (e.key !== "Tab") return;
    const focusables = modal.querySelectorAll(
      'button:not([disabled]):not([hidden]), textarea:not([disabled]), a[href], input:not([disabled])'
    );
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  });

  readUrl();
  load();
})();
