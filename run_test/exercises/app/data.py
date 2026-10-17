"""数据层：加载中文动作数据、派生字段、筛选与媒体白名单。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_FILE = ROOT / "exercises-dataset" / "data" / "exercises_zh.json"
MEDIA_DIR = ROOT / "exercises-dataset" / "videos"


def _load() -> list[dict[str, Any]]:
    rows = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        gif_url = str(item.get("gif_url") or "")
        item["gif"] = gif_url.split("/")[-1] if gif_url else ""
        items.append(item)
    return items


EXERCISES: list[dict[str, Any]] = _load()
BY_ID: dict[str, dict[str, Any]] = {item["id"]: item for item in EXERCISES}
TOTAL = len(EXERCISES)


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({v for v in values if v})


def facets() -> dict[str, list[str]]:
    """返回筛选栏所需的去重排序取值。"""
    return {
        "categories": _sorted_unique([e.get("category", "") for e in EXERCISES]),
        "targets": _sorted_unique([e.get("target", "") for e in EXERCISES]),
        "equipments": _sorted_unique([e.get("equipment", "") for e in EXERCISES]),
    }


def filter_exercises(
    category: str = "",
    target: str = "",
    equipment: str = "",
    q: str = "",
) -> list[dict[str, Any]]:
    """多条件 AND 筛选；空值表示不过滤；q 对名称做不区分大小写包含匹配。"""
    keyword = (q or "").strip().lower()
    result = []
    for item in EXERCISES:
        if category and item.get("category") != category:
            continue
        if target and item.get("target") != target:
            continue
        if equipment and item.get("equipment") != equipment:
            continue
        if keyword and keyword not in str(item.get("name", "")).lower():
            continue
        result.append(item)
    return result


def summary(item: dict[str, Any]) -> dict[str, Any]:
    """列表接口使用的精简字段。"""
    return {
        "id": item["id"],
        "name": item.get("name", ""),
        "category": item.get("category", ""),
        "body_part": item.get("body_part", ""),
        "equipment": item.get("equipment", ""),
        "target": item.get("target", ""),
        "muscle_group": item.get("muscle_group", ""),
        "gif": item.get("gif", ""),
    }


def get_exercise(exercise_id: str) -> dict[str, Any] | None:
    return BY_ID.get(exercise_id)


def media_path(name: str) -> Path | None:
    """只允许 videos 目录内已存在的 .gif 文件，拒绝路径穿越。"""
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    if not name.lower().endswith(".gif"):
        return None
    path = MEDIA_DIR / name
    if not path.is_file():
        return None
    return path
