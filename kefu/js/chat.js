// 对话交互模块
class ChatManager {
  constructor(productManager, recommendEngine) {
    this.productManager = productManager;
    this.recommendEngine = recommendEngine;
    this.conversationHistory = [];
    this.greetingMessage = '您好！欢迎来到手机专卖店智能客服。我可以帮您：\n\n1. 📱 查询手机信息和价格\n2. 💰 根据预算推荐手机\n3. 🔧 解答售后问题\n\n请问有什么可以帮您的吗？';
  }

  // 处理用户输入
  async processInput(input) {
    const userMessage = input.trim();
    if (!userMessage) return '';

    // 记录用户消息
    this.addToHistory('user', userMessage);

    // 分析用户意图并生成回复
    const response = await this.generateResponse(userMessage);

    // 记录客服回复
    this.addToHistory('assistant', response);

    return response;
  }

  // 生成回复
  async generateResponse(input) {
    const lowerInput = input.toLowerCase();

    // 问候语
    if (this.isGreeting(lowerInput)) {
      return this.greetingMessage;
    }

    // 预算推荐
    if (this.isBudgetQuery(lowerInput)) {
      return this.handleBudgetQuery(input);
    }

    // 价格查询放在预算推荐之后，避免“推荐3000元左右”被误判为价格查询。
    if (this.isPriceQuery(lowerInput)) {
      return this.handlePriceQuery(input);
    }

    // 品牌查询
    if (this.isBrandQuery(lowerInput)) {
      return this.handleBrandQuery(input);
    }

    // 售后问题
    if (this.isAfterSalesQuery(lowerInput)) {
      return this.handleAfterSalesQuery(input);
    }

    // 对比查询
    if (this.isComparisonQuery(lowerInput)) {
      return this.handleComparisonQuery(input);
    }

    // 通用搜索
    return this.handleGeneralSearch(input);
  }

  // 判断是否为问候语
  isGreeting(input) {
    const greetings = ['你好', '您好', 'hi', 'hello', '在吗', '在不在', '嗨'];
    return greetings.some(g => input.includes(g));
  }

  // 判断是否为价格查询
  isPriceQuery(input) {
    const priceKeywords = ['多少钱', '价格', '售价', '什么价', '价位', '元'];
    return priceKeywords.some(k => input.includes(k));
  }

  // 判断是否为预算查询
  isBudgetQuery(input) {
    const budgetKeywords = ['推荐', '预算', '左右', '以内', '以下', '以内', '性价比'];
    return budgetKeywords.some(k => input.includes(k));
  }

  // 判断是否为品牌查询
  isBrandQuery(input) {
    const brands = this.productManager.getBrands().map(b => b.toLowerCase());
    return brands.some(b => input.toLowerCase().includes(b));
  }

  // 判断是否为售后问题
  isAfterSalesQuery(input) {
    const afterSalesKeywords = ['售后', '保修', '退换', '维修', '换货', '退货', '质保', '服务'];
    return afterSalesKeywords.some(k => input.includes(k));
  }

  // 判断是否为对比查询
  isComparisonQuery(input) {
    const compareKeywords = ['对比', '比较', '区别', '差异', '哪个好', 'vs'];
    return compareKeywords.some(k => input.includes(k));
  }

  // 处理价格查询
  handlePriceQuery(input) {
    const phones = this.productManager.searchByKeyword(input);
    
    if (phones.length === 0) {
      return '抱歉，没有找到相关的手机信息。您可以告诉我具体的品牌或型号，比如"iPhone 15"或"华为Mate 60"。';
    }

    let response = '为您查询到以下手机信息：\n\n';
    phones.slice(0, 3).forEach(phone => {
      response += `📱 **${phone.brand} ${phone.model}**\n`;
      response += `   价格：¥${phone.price.toLocaleString()}\n`;
      response += `   可选颜色：${phone.color.join('、')}\n`;
      response += `   存储容量：${phone.storage.join('、')}\n\n`;
    });

    return response;
  }

  // 处理预算查询
  handleBudgetQuery(input) {
    // 提取预算金额
    const budget = this.extractBudget(input);
    
    if (!budget) {
      return '请告诉我您的预算范围，比如"3000元左右"或"5000元以内"，我来为您推荐合适的手机。';
    }

    const recommendations = this.recommendEngine.recommendByBudget(budget);
    return this.recommendEngine.generateRecommendationResponse(
      recommendations,
      `根据您的预算 ¥${budget.toLocaleString()}，`
    );
  }

  // 提取预算金额
  extractBudget(input) {
    // 匹配"3000元"、"3000左右"、"3000以内"等格式
    const patterns = [
      /(\d+)\s*元?\s*(?:左右|上下)/,
      /(\d+)\s*元?\s*(?:以内|以下|之内)/,
      /(\d+)\s*元?\s*(?:以上|之上)/,
      /预算\s*(\d+)/,
      /(\d+)\s*元/
    ];

    for (const pattern of patterns) {
      const match = input.match(pattern);
      if (match) {
        return parseInt(match[1]);
      }
    }
    return null;
  }

  // 处理品牌查询
  handleBrandQuery(input) {
    const brands = this.productManager.getBrands();
    const foundBrand = brands.find(b => input.includes(b));
    
    if (!foundBrand) {
      return '我们提供以下品牌的手机：苹果、华为、小米、OPPO、vivo、荣耀、三星、一加、realme。请问您对哪个品牌感兴趣？';
    }

    const phones = this.productManager.searchByBrand(foundBrand);
    let response = `为您展示 ${foundBrand} 品牌的手机：\n\n`;
    
    phones.forEach(phone => {
      response += `📱 **${phone.model}** - ¥${phone.price.toLocaleString()}\n`;
      response += `   ${phone.screen} | ${phone.processor}\n\n`;
    });

    return response;
  }

  // 处理售后问题
  handleAfterSalesQuery(input) {
    const lowerInput = input.toLowerCase();

    // 退换货政策
    if (lowerInput.includes('退') || lowerInput.includes('换')) {
      return '📋 **退换货政策**\n\n' +
        '1. 自签收之日起7天内，如商品无人为损坏，可申请无理由退货\n' +
        '2. 15天内出现质量问题，可选择换货或维修\n' +
        '3. 退换货时请保持商品包装完好，配件齐全\n' +
        '4. 激活后的手机不支持无理由退货，但享受质量保障\n\n' +
        '如有需要，请联系我们的客服热线：400-XXX-XXXX';
    }

    // 保修政策
    if (lowerInput.includes('保修') || lowerInput.includes('质保')) {
      return '🔧 **保修政策**\n\n' +
        '1. 所有手机均提供1年官方保修服务\n' +
        '2. 保修范围：非人为损坏的性能故障\n' +
        '3. 保修期内免费维修，需提供购买凭证\n' +
        '4. 电池保修期为6个月\n' +
        '5. 可额外购买延保服务，延长至2-3年\n\n' +
        '您也可以在购买时询问店员了解详细的保修条款。';
    }

    // 维修服务
    if (lowerInput.includes('维修')) {
      return '🛠️ **维修服务**\n\n' +
        '1. 保修期内：免费维修（非人为损坏）\n' +
        '2. 保修期外：收取相应维修费用\n' +
        '3. 维修时间：一般3-7个工作日\n' +
        '4. 提供备用机服务（部分机型）\n\n' +
        '建议您到店或联系客服热线获取更详细的维修信息。';
    }

    // 通用售后回复
    return '📞 **售后服务**\n\n' +
      '我们提供以下售后服务：\n' +
      '• 7天无理由退货\n' +
      '• 15天质量问题换货\n' +
      '• 1年官方保修\n' +
      '• 终身技术支持\n\n' +
      '请问您具体想了解哪方面的售后政策？';
  }

  // 处理对比查询
  handleComparisonQuery(input) {
    const phones = this.productManager.searchByKeyword(input);
    
    if (phones.length < 2) {
      return '请告诉我您想对比的两款手机型号，比如"iPhone 15和华为Mate 60对比"。';
    }

    const phone1 = phones[0];
    const phone2 = phones[1];

    let response = `📊 **${phone1.brand} ${phone1.model}** vs **${phone2.brand} ${phone2.model}**\n\n`;
    response += `| 项目 | ${phone1.model} | ${phone2.model} |\n`;
    response += `|------|----------------|----------------|\n`;
    response += `| 价格 | ¥${phone1.price.toLocaleString()} | ¥${phone2.price.toLocaleString()} |\n`;
    response += `| 处理器 | ${phone1.processor} | ${phone2.processor} |\n`;
    response += `| 屏幕 | ${phone1.screen} | ${phone2.screen} |\n`;
    response += `| 电池 | ${phone1.battery} | ${phone2.battery} |\n`;
    response += `| 特色 | ${phone1.features.slice(0, 2).join('、')} | ${phone2.features.slice(0, 2).join('、')} |\n\n`;
    
    response += '您可以告诉我更具体的对比需求，或者询问某款手机的详细信息。';
    return response;
  }

  // 通用搜索
  handleGeneralSearch(input) {
    const phones = this.productManager.searchByKeyword(input);
    
    if (phones.length > 0) {
      let response = '为您找到以下相关手机：\n\n';
      phones.slice(0, 3).forEach(phone => {
        response += `📱 **${phone.brand} ${phone.model}** - ¥${phone.price.toLocaleString()}\n`;
        response += `   ${phone.screen} | ${phone.processor}\n\n`;
      });
      return response;
    }

    return '抱歉，我没有理解您的问题。您可以尝试询问：\n\n' +
      '• "iPhone 15多少钱" - 查询价格\n' +
      '• "推荐3000元左右的手机" - 预算推荐\n' +
      '• "华为手机有哪些" - 品牌查询\n' +
      '• "退换货政策" - 售后问题\n\n' +
      '请问有什么可以帮您的？';
  }

  // 添加到历史记录
  addToHistory(role, content) {
    this.conversationHistory.push({
      role,
      content,
      timestamp: new Date().toISOString()
    });

    // 限制历史记录长度
    if (this.conversationHistory.length > 50) {
      this.conversationHistory = this.conversationHistory.slice(-50);
    }
  }

  // 获取历史记录
  getHistory() {
    return this.conversationHistory;
  }

  // 清空历史记录
  clearHistory() {
    this.conversationHistory = [];
  }
}
