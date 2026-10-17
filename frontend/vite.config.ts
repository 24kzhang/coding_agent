// 作者：张又升
// 邮箱：1041586605@qq.com
// 单位：西北大学 电子信息学院(人工智能学院)

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    strictPort: false
  }
});
