// 健身动作网站前端逻辑

const PAGE_SIZE = 24;
let currentPage = 1;
let totalPages = 1;
let currentFilters = { category: '', muscle: '', equipment: '' };
let allOptions = { categories: [], muscles: [], equipment: [] };

// 当前打开的动作 ID
let currentExerciseId = null;
// 每个动作独立的对话历史 Map<exerciseId, Array<{role, content}>>
const exerciseChatHistories = new Map();
// 每个动作是否已经发送过首次消息
const exerciseFirstSent = new Map();

// HTML 转义
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const str = String(text);
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return str.replace(/[&<>"']/g, c => map[c]);
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    loadOptions();
    loadExercises();
    bindEvents();
});

// 绑定事件
function bindEvents() {
    document.getElementById('filter-category').addEventListener('change', onFilterChange);
    document.getElementById('filter-muscle').addEventListener('change', onFilterChange);
    document.getElementById('filter-equipment').addEventListener('change', onFilterChange);
    document.getElementById('btn-reset').addEventListener('click', resetFilters);
    document.getElementById('btn-prev').addEventListener('click', () => goToPage(currentPage - 1));
    document.getElementById('btn-next').addEventListener('click', () => goToPage(currentPage + 1));
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.getElementById('modal-overlay').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeModal();
    });
    document.getElementById('chat-send').addEventListener('click', sendChatMessage);
    document.getElementById('chat-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}

// 重置筛选
function resetFilters() {
    document.getElementById('filter-category').value = '';
    document.getElementById('filter-muscle').value = '';
    document.getElementById('filter-equipment').value = '';
    currentFilters = { category: '', muscle: '', equipment: '' };
    currentPage = 1;
    loadExercises();
}

// 加载筛选选项
async function loadOptions() {
    try {
        const res = await fetch('/api/options');
        allOptions = await res.json();
        fillSelect('filter-category', allOptions.categories || [], '全部分类');
        fillSelect('filter-muscle', allOptions.muscles || [], '全部目标肌肉');
        fillSelect('filter-equipment', allOptions.equipment || [], '全部器材');
    } catch (e) {
        console.error('加载选项失败', e);
    }
}

function fillSelect(id, items, placeholder) {
    const sel = document.getElementById(id);
    sel.innerHTML = `<option value="">${escapeHtml(placeholder)}</option>`;
    items.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item;
        opt.textContent = item;
        sel.appendChild(opt);
    });
}

// 筛选变更
function onFilterChange() {
    currentFilters.category = document.getElementById('filter-category').value;
    currentFilters.muscle = document.getElementById('filter-muscle').value;
    currentFilters.equipment = document.getElementById('filter-equipment').value;
    currentPage = 1;
    loadExercises();
}

// 加载动作列表
async function loadExercises() {
    const params = new URLSearchParams({ page: currentPage });
    if (currentFilters.category) params.set('category', currentFilters.category);
    if (currentFilters.muscle) params.set('muscle', currentFilters.muscle);
    if (currentFilters.equipment) params.set('equipment', currentFilters.equipment);

    try {
        const res = await fetch(`/api/exercises?${params}`);
        const data = await res.json();
        totalPages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));
        renderGrid(data.items || []);
        renderPagination(data.total || 0);
    } catch (e) {
        console.error('加载动作失败', e);
    }
}

// 渲染卡片网格
function renderGrid(items) {
    const grid = document.getElementById('exercise-grid');
    if (!items.length) {
        grid.innerHTML = '<p style="text-align:center;color:#999;padding:40px;">没有找到匹配的动作</p>';
        return;
    }

    grid.innerHTML = items.map(ex => `
        <div class="exercise-card" data-id="${escapeHtml(ex.id)}">
            <div class="card-image">
                <img src="/media/${escapeHtml(ex.gif_url || '')}" alt="${escapeHtml(ex.name)}" loading="lazy">
            </div>
            <div class="card-body">
                <h3 class="card-title">${escapeHtml(ex.name)}</h3>
                <div class="card-meta">
                    <span class="tag">部位: ${escapeHtml(ex.body_part || '')}</span>
                    <span class="tag">器材: ${escapeHtml(ex.equipment || '')}</span>
                    <span class="tag">目标: ${escapeHtml(ex.target || '')}</span>
                </div>
            </div>
        </div>
    `).join('');

    // 绑定卡片点击事件
    grid.querySelectorAll('.exercise-card').forEach(card => {
        card.addEventListener('click', () => openDetail(card.dataset.id));
    });
}

// 渲染分页
function renderPagination(total) {
    document.getElementById('total-count').textContent = total;
    document.getElementById('page-info').textContent = `第 ${currentPage} / ${totalPages} 页`;
    document.getElementById('btn-prev').disabled = currentPage <= 1;
    document.getElementById('btn-next').disabled = currentPage >= totalPages;
}

// 翻页
function goToPage(page) {
    if (page < 1 || page > totalPages) return;
    currentPage = page;
    loadExercises();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// 打开详情弹窗
async function openDetail(exerciseId) {
    currentExerciseId = exerciseId;
    try {
        const res = await fetch(`/api/exercises/${encodeURIComponent(exerciseId)}`);
        if (!res.ok) throw new Error('加载失败');
        const ex = await res.json();
        populateDetail(ex);
        document.getElementById('modal-overlay').classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        loadChatHistory(exerciseId);
    } catch (e) {
        console.error('加载详情失败', e);
    }
}

// 填充详情
function populateDetail(ex) {
    document.getElementById('detail-gif').src = `/media/${ex.gif_url || ''}`;
    document.getElementById('detail-name').textContent = ex.name || '';
    document.getElementById('detail-category').textContent = ex.category || '';
    document.getElementById('detail-body-part').textContent = ex.body_part || '';
    document.getElementById('detail-target').textContent = ex.target || '';
    document.getElementById('detail-muscle-group').textContent = ex.muscle_group || '';
    document.getElementById('detail-secondary').textContent = (ex.secondary_muscles || []).join('、');
    document.getElementById('detail-equipment').textContent = ex.equipment || '';
    document.getElementById('detail-instructions').textContent = (ex.instructions && ex.instructions.zh) || '';

    const stepsEl = document.getElementById('detail-steps');
    const steps = (ex.instruction_steps && ex.instruction_steps.zh) || [];
    stepsEl.innerHTML = steps.map(s => `<li>${escapeHtml(s)}</li>`).join('');
}

// 关闭弹窗
function closeModal() {
    document.getElementById('modal-overlay').classList.add('hidden');
    document.body.style.overflow = '';
    currentExerciseId = null;
}

// 加载对话历史到界面
function loadChatHistory(exerciseId) {
    const container = document.getElementById('chat-messages');
    container.innerHTML = '';
    const history = exerciseChatHistories.get(exerciseId) || [];
    history.forEach(msg => {
        appendChatBubble(msg.role, msg.content);
    });
    if (!history.length) {
        appendChatBubble('assistant', '你好！我是健身助手，关于这个动作有什么问题都可以问我哦 😊');
    }
}

// 添加聊天气泡
function appendChatBubble(role, content) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    div.innerHTML = `<p>${escapeHtml(content)}</p>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

// 添加打字指示器
function addTypingIndicator() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'chat-msg assistant';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

// 发送聊天消息（SSE 流式）
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message || !currentExerciseId) return;

    // 记录用户消息
    if (!exerciseChatHistories.has(currentExerciseId)) {
        exerciseChatHistories.set(currentExerciseId, []);
    }
    const history = exerciseChatHistories.get(currentExerciseId);
    history.push({ role: 'user', content: message });
    const userBubble = appendChatBubble('user', message);
    input.value = '';

    // 禁用输入
    const sendBtn = document.getElementById('chat-send');
    sendBtn.disabled = true;
    input.disabled = true;
    addTypingIndicator();

    // 是否为首次发送（在请求前记录，用于决定是否传 is_first）
    const isFirst = !exerciseFirstSent.get(currentExerciseId);
    let assistantBubble = null;

    try {
        const body = {
            message: message,
            history: isFirst ? [] : history.slice(0, -1),
            is_first: isFirst
        };

        const res = await fetch(`/api/chat/${encodeURIComponent(currentExerciseId)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (!res.ok) throw new Error(`请求失败: ${res.status}`);

        removeTypingIndicator();

        // 读取 SSE 流
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let assistantContent = '';
        assistantBubble = appendChatBubble('assistant', '');
        const pEl = assistantBubble.querySelector('p');
        let buffer = '';
        let hasError = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // 解析 SSE 事件
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    try {
                        const parsed = JSON.parse(data);
                        // 优先处理错误信息
                        if (parsed.error) {
                            assistantContent = String(parsed.error);
                            pEl.textContent = assistantContent;
                            hasError = true;
                            break;
                        }
                        if (parsed.content) {
                            assistantContent += parsed.content;
                            pEl.textContent = assistantContent;
                        }
                    } catch (e) {
                        // 忽略解析错误
                    }
                }
            }
            if (hasError) break;
            const container = document.getElementById('chat-messages');
            container.scrollTop = container.scrollHeight;
        }

        // 仅在无错误且助手有非空内容时记录历史并标记首次已发送
        if (!hasError && assistantContent) {
            history.push({ role: 'assistant', content: assistantContent });
            if (isFirst) exerciseFirstSent.set(currentExerciseId, true);
        } else {
            // 失败（SSE 错误、空回复）：回滚到首次状态
            history.pop(); // 移除本次用户消息
            userBubble.remove(); // 移除用户气泡
            if (assistantBubble) assistantBubble.remove(); // 移除临时助手气泡
            // 不设置 exerciseFirstSent，使重试仍传 is_first=true
        }

    } catch (e) {
        removeTypingIndicator();
        // HTTP 错误或读取异常也回滚到首次状态
        history.pop();
        userBubble.remove();
        if (assistantBubble) assistantBubble.remove();
        console.error('聊天请求失败', e);
    } finally {
        sendBtn.disabled = false;
        input.disabled = false;
        input.focus();
    }
}