"""数据层与 HTTP 接口测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import data
import main

client = TestClient(main.app)
RAW = json.loads(data.DATA_FILE.read_text(encoding="utf-8"))


def test_total_matches_file() -> None:
    assert data.TOTAL == len(RAW) == 1324
    res = client.get("/api/exercises")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == len(RAW)
    assert body["count"] == len(RAW)
    assert len(body["items"]) == len(RAW)


def test_facets_non_empty() -> None:
    f = data.facets()
    assert f["categories"] and f["targets"] and f["equipments"]
    assert f["categories"] == sorted(set(f["categories"]))
    assert "腰部" in f["categories"]


def test_combined_filter_is_and() -> None:
    items = data.filter_exercises(category="腰部", target="腹肌", equipment="自重")
    assert items
    for item in items:
        assert item["category"] == "腰部"
        assert item["target"] == "腹肌"
        assert item["equipment"] == "自重"
    expected = [
        r for r in RAW
        if r["category"] == "腰部" and r["target"] == "腹肌" and r["equipment"] == "自重"
    ]
    assert len(items) == len(expected)


def test_filter_clear_restores_all() -> None:
    assert len(data.filter_exercises()) == data.TOTAL
    assert len(data.filter_exercises(category="", target="", equipment="", q="")) == data.TOTAL


def test_search_by_name() -> None:
    items = data.filter_exercises(q="仰卧起坐")
    assert items
    assert all("仰卧起坐" in item["name"] for item in items)


def test_detail_has_all_raw_fields() -> None:
    res = client.get("/api/exercises/0001")
    assert res.status_code == 200
    body = res.json()
    raw = next(r for r in RAW if r["id"] == "0001")
    for key in raw:
        assert key in body
    assert body["gif"] == raw["gif_url"].split("/")[-1]
    assert body["gif"].endswith(".gif")


def test_unknown_id_404() -> None:
    res = client.get("/api/exercises/9999")
    assert res.status_code == 404
    assert res.json()["detail"] == "未找到该动作"


def test_media_served_and_traversal_rejected() -> None:
    name = RAW[0]["gif_url"].split("/")[-1]
    ok = client.get(f"/media/{name}")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "image/gif"
    for bad in ("../data/exercises_zh.json", "..%2Fdata%2Fexercises_zh.json", "0001.jpg", "nope.gif"):
        assert client.get(f"/media/{bad}").status_code == 404
    assert data.media_path("../x.gif") is None
    assert data.media_path("0001-2gPfomN.jpg") is None


def test_index_page_renders() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "动作库" in res.text
    assert "/static/app.js" in res.text
    assert "images/" not in res.text


def test_all_gif_files_exist() -> None:
    missing = [r["id"] for r in RAW if not (data.MEDIA_DIR / r["gif_url"].split("/")[-1]).is_file()]
    assert missing == []
    assert Path(data.MEDIA_DIR).is_dir()
