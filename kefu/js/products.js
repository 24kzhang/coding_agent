// 商品数据管理模块
class ProductManager {
  constructor() {
    this.phones = [];
    this.initialized = false;
  }

  // 初始化加载商品数据
  async init() {
    try {
      const response = await fetch('data/phones.json');
      const data = await response.json();
      this.phones = data.phones;
      this.initialized = true;
      console.log(`已加载 ${this.phones.length} 款手机数据`);
    } catch (error) {
      console.error('加载商品数据失败:', error);
      this.phones = [];
    }
  }

  // 获取所有商品
  getAllPhones() {
    return this.phones;
  }

  // 根据ID获取商品
  getPhoneById(id) {
    return this.phones.find(phone => phone.id === id);
  }

  // 根据品牌搜索
  searchByBrand(brand) {
    return this.phones.filter(phone => 
      phone.brand.toLowerCase().includes(brand.toLowerCase())
    );
  }

  // 根据型号搜索
  searchByModel(model) {
    return this.phones.filter(phone => 
      phone.model.toLowerCase().includes(model.toLowerCase())
    );
  }

  // 根据价格范围筛选
  filterByPriceRange(minPrice, maxPrice) {
    return this.phones.filter(phone => 
      phone.price >= minPrice && phone.price <= maxPrice
    );
  }

  // 根据关键词搜索（品牌、型号、特性）
  searchByKeyword(keyword) {
    const lowerKeyword = keyword.toLowerCase();
    const compactKeyword = lowerKeyword.replace(/\s+/g, '');

    return this.phones.filter(phone => {
      const brand = phone.brand.toLowerCase();
      const model = phone.model.toLowerCase();
      const compactModel = model.replace(/\s+/g, '');
      const processor = phone.processor.toLowerCase();

      return (
        lowerKeyword.includes(brand) ||
        compactKeyword.includes(compactModel) ||
        model.includes(lowerKeyword) ||
        phone.features.some(f => lowerKeyword.includes(f.toLowerCase()) || f.toLowerCase().includes(lowerKeyword)) ||
        lowerKeyword.includes(processor) ||
        processor.includes(lowerKeyword)
      );
    });
  }

  // 获取价格区间统计
  getPriceRange() {
    if (this.phones.length === 0) return { min: 0, max: 0 };
    const prices = this.phones.map(p => p.price);
    return {
      min: Math.min(...prices),
      max: Math.max(...prices)
    };
  }

  // 获取所有品牌列表
  getBrands() {
    return [...new Set(this.phones.map(p => p.brand))];
  }

  // 格式化价格显示
  formatPrice(price) {
    return `¥${price.toLocaleString()}`;
  }

  // 获取商品简要信息
  getPhoneSummary(phone) {
    return {
      id: phone.id,
      name: `${phone.brand} ${phone.model}`,
      price: this.formatPrice(phone.price),
      screen: phone.screen,
      processor: phone.processor,
      camera: phone.camera,
      battery: phone.battery
    };
  }
}

// 创建全局实例
const productManager = new ProductManager();
