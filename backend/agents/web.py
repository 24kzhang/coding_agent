# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

import re
from html.parser import HTMLParser


class _Dialogs(HTMLParser):
    """收集非原生模态的定位标识，普通脚本定位类不参与样式验收。"""

    def __init__(self) -> None:
        """初始化当前对象及其运行状态。"""
        super().__init__()
        self.dialogs: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        """收集网页起始标签中的相关信息。"""
        values = dict(attrs)
        if tag != "dialog" and values.get("role") == "dialog" and values.get("aria-modal") == "true":
            self.dialogs.append(values)


def dialog_style_issues(html: str, css: str, js: str) -> list[str]:
    """静态 HTML 的自定义模态必须有可定位的样式，运行效果仍需浏览器验证。"""

    parser = _Dialogs()
    parser.feed(html)
    if re.search(r'<(?:link|script)[^>]+(?:href|src)=["\']https?://', html) or "insertRule(" in js:
        return []
    clean_css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    selectors = "\n".join(re.findall(r"([^{}]+)\{", clean_css))
    issues = []
    for attrs in parser.dialogs:
        tokens = ["." + name for name in str(attrs.get("class") or "").split()]
        if attrs.get("id"):
            tokens.append("#" + attrs["id"])
        covered = any(re.search(re.escape(token) + r"(?![\w-])", selectors) for token in tokens)
        covered = covered or bool(re.search(r'\[role\s*=\s*["\']?dialog', selectors))
        covered = covered or bool(attrs.get("style"))
        if not covered:
            label = attrs.get("id") or attrs.get("class") or "未命名"
            issues.append(f"自定义模态样式缺失：{label}，aria-modal 容器没有匹配的本地样式；必须补齐定位、遮罩和布局并验证实际窗口")
    return issues
