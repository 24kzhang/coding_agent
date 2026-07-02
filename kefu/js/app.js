// 主应用逻辑
class App {
  constructor() {
    this.productManager = productManager;
    this.recommendEngine = null;
    this.chatManager = null;
    this.isInitialized = false;
  }

  // 初始化应用
  async init() {
    try {
      // 加载商品数据
      await this.productManager.init();
      
      // 初始化推荐引擎
      this.recommendEngine = initRecommendEngine(this.productManager);
      
      // 初始化聊天管理器
      this.chatManager = new ChatManager(this.productManager, this.recommendEngine);
      
      // 绑定UI事件
      this.bindEvents();
      
      // 显示欢迎消息
      this.showWelcomeMessage();
      
      // 渲染热门机型
      this.renderHotProducts();
      
      this.isInitialized = true;
      console.log('应用初始化完成');
    } catch (error) {
      console.error('应用初始化失败:', error);
      this.showErrorMessage('系统加载失败，请刷新页面重试');
    }
  }

  // 绑定UI事件
  bindEvents() {
    const input = document.getElementById('userInput');
    const sendBtn = document.querySelector('.chat-input-area button');

    // 发送按钮点击事件
    if (sendBtn) {
      sendBtn.addEventListener('click', () => this.sendMessage());
    }
    
    // 输入框回车发送
    if (input) {
      input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
    }

    // 快捷问题由 HTML 上的全局函数处理，避免重复绑定导致发送两次。
  }

  // 发送消息
  async sendMessage() {
    const input = document.getElementById('userInput');
    if (!input) return;
    
    const message = input.value.trim();
    if (!message) return;

    // 显示用户消息
    this.addMessage(message, 'user');
    input.value = '';

    // 显示加载状态
    this.showLoading();

    try {
      // 处理消息并获取回复
      const response = await this.chatManager.processInput(message);
      
      // 隐藏加载状态
      this.hideLoading();
      
      // 显示客服回复
      this.addMessage(response, 'assistant');
    } catch (error) {
      console.error('处理消息失败:', error);
      this.hideLoading();
      this.addMessage('抱歉，处理您的请求时出现了错误，请稍后重试。', 'assistant');
    }
  }

  // 发送快捷问题
  sendQuickQuestion(question) {
    const input = document.getElementById('userInput');
    if (input) {
      input.value = question;
    }
    this.sendMessage();
  }

  // 添加消息到对话区域
  addMessage(content, type) {
    const chatArea = document.getElementById('chatMessages');
    if (!chatArea) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = this.formatMessage(content);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatArea.appendChild(messageDiv);
    
    // 滚动到底部
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // 格式化消息内容
  formatMessage(content) {
    // 将换行符转换为<br>
    let formatted = content.replace(/\n/g, '<br>');
    
    // 将**文本**转换为<strong>文本</strong>
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    
    return formatted;
  }

  // 显示加载状态
  showLoading() {
    const chatArea = document.getElementById('chatMessages');
    if (!chatArea) return;
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant-message loading';
    loadingDiv.id = 'loading-message';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<span class="loading-dots">正在输入...</span>';
    
    loadingDiv.appendChild(avatar);
    loadingDiv.appendChild(contentDiv);
    chatArea.appendChild(loadingDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // 隐藏加载状态
  hideLoading() {
    const loading = document.getElementById('loading-message');
    if (loading) {
      loading.remove();
    }
  }

  // 显示欢迎消息
  showWelcomeMessage() {
    const welcome = this.chatManager.greetingMessage;
    this.addMessage(welcome, 'assistant');
  }

  // 显示错误消息
  showErrorMessage(message) {
    const chatArea = document.getElementById('chatMessages');
    if (chatArea) {
      chatArea.innerHTML = `<div class="error-message">${message}</div>`;
    }
  }

  // 渲染热门机型
  renderHotProducts() {
    const hotProductsList = document.getElementById('hotProductsList');
    if (!hotProductsList) return;
    
    const phones = this.productManager.getAllPhones().slice(0, 5);
    
    phones.forEach(phone => {
      const item = document.createElement('div');
      item.className = 'hot-product-item';
      item.innerHTML = `
        <div>${phone.brand} ${phone.model}</div>
        <div class="price">¥${phone.price.toLocaleString()}</div>
      `;
      item.addEventListener('click', () => {
        const input = document.getElementById('userInput');
        if (input) {
          input.value = `${phone.brand} ${phone.model}多少钱`;
          this.sendMessage();
        }
      });
      hotProductsList.appendChild(item);
    });
  }
}

// 全局函数，供 HTML 内联事件调用
function sendMessage() {
  if (window.app) {
    window.app.sendMessage();
  }
}

function sendQuickQuestion(question) {
  if (window.app) {
    window.app.sendQuickQuestion(question);
  }
}

function handleKeyPress(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
  window.app.init();
});
