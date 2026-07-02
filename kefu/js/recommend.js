// 推荐算法模块
class RecommendEngine {
  constructor(productManager) {
    this.productManager = productManager;
  }

  // 根据预算推荐手机
  recommendByBudget(budget, tolerance = 500) {
    const phones = this.productManager.getAllPhones();
    const minBudget = budget - tolerance;
    const maxBudget = budget + tolerance;
    
    // 筛选符合预算的手机
    const candidates = phones.filter(phone => 
      phone.price >= minBudget && phone.price <= maxBudget
    );

    if (candidates.length === 0) {
      // 如果没有精确匹配，找最接近的
      return this.findClosestByBudget(budget);
    }

    // 按性价比排序（这里简化为按价格接近程度排序）
    return candidates
      .sort((a, b) => Math.abs(a.price - budget) - Math.abs(b.price - budget))
      .slice(0, 3);
  }

  // 找最接近预算的手机
  findClosestByBudget(budget) {
    const phones = this.productManager.getAllPhones();
    return [...phones]
      .sort((a, b) => Math.abs(a.price - budget) - Math.abs(b.price - budget))
      .slice(0, 3);
  }

  // 根据需求关键词推荐
  recommendByNeeds(keywords) {
    const phones = this.productManager.getAllPhones();
    const lowerKeywords = keywords.map(k => k.toLowerCase());
    
    // 为每个手机计算匹配分数
    const scored = phones.map(phone => {
      let score = 0;
      const phoneText = [
        phone.brand,
        phone.model,
        phone.processor,
        phone.camera,
        ...phone.features
      ].join(' ').toLowerCase();

      lowerKeywords.forEach(keyword => {
        if (phoneText.includes(keyword)) {
          score += 10;
        }
      });

      return { phone, score };
    });

    return scored
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map(item => item.phone);
  }

  // 根据品牌偏好推荐
  recommendByBrand(brand) {
    return this.productManager.searchByBrand(brand);
  }

  // 获取性价比推荐（中等价位）
  getBestValueRecommendations() {
    const phones = this.productManager.getAllPhones();
    const priceRange = this.productManager.getPriceRange();
    const midPrice = (priceRange.min + priceRange.max) / 2;
    
    return phones
      .filter(phone => phone.price >= midPrice * 0.7 && phone.price <= midPrice * 1.3)
      .sort((a, b) => b.price - a.price)
      .slice(0, 5);
  }

  // 生成推荐回复文本
  generateRecommendationResponse(recommendations, context = '') {
    if (recommendations.length === 0) {
      return '抱歉，没有找到符合您需求的手机。您可以告诉我更具体的需求，比如预算范围或品牌偏好。';
    }

    let response = '';
    if (context) {
      response += `${context}\n\n`;
    }
    response += `为您推荐以下 ${recommendations.length} 款手机：\n\n`;

    recommendations.forEach((phone, index) => {
      response += `${index + 1}. **${phone.brand} ${phone.model}**\n`;
      response += `   - 价格：¥${phone.price.toLocaleString()}\n`;
      response += `   - 处理器：${phone.processor}\n`;
      response += `   - 屏幕：${phone.screen}\n`;
      response += `   - 电池：${phone.battery}\n`;
      response += `   - 特色：${phone.features.slice(0, 3).join('、')}\n\n`;
    });

    response += '您可以点击了解更多详情，或者告诉我您的其他需求。';
    return response;
  }
}

// 创建全局实例（需要在productManager初始化后）
let recommendEngine = null;

function initRecommendEngine(productMgr) {
  recommendEngine = new RecommendEngine(productMgr);
  return recommendEngine;
}
