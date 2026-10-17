# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

from datetime import UTC, datetime


def utc_stamp() -> str:
    """返回紧凑、可排序、可被浏览器直接解析的 UTC 时间。

    JSONL 记忆不需要微秒和 ``+00:00`` 偏移信息，因此统一保存为
    ``2026-07-18T09:30:12Z``。相比 ``datetime.isoformat()`` 更短，同时仍然
    保留秒级精度和明确时区。
    """

    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")
