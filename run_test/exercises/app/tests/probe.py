"""数据探查脚本：打印中文数据集的字段与取值分布，供实现前核对。"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "exercises-dataset" / "data" / "exercises_zh.json"
VIDEOS = ROOT / "exercises-dataset" / "videos"


def main() -> None:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    print("总条数:", len(rows))
    print("字段集合:", sorted({k for r in rows for k in r}))
    print("instructions 键集合:", sorted({k for r in rows for k in r.get("instructions", {})}))
    print("instruction_steps 键集合:", sorted({k for r in rows for k in r.get("instruction_steps", {})}))
    for field in ("category", "body_part", "equipment", "target"):
        values = Counter(r.get(field) for r in rows)
        print(f"{field} 取值({len(values)}):", dict(values))
    prefixes = Counter(str(r.get("gif_url", "")).split("/")[0] for r in rows)
    print("gif_url 前缀分布:", dict(prefixes))
    gifs = {p.name for p in VIDEOS.glob("*.gif")}
    refs = {str(r.get("gif_url", "")).split("/")[-1] for r in rows}
    print("videos 目录 gif 数:", len(gifs), "gif_url 引用数:", len(refs), "一致:", gifs == refs)
    missing = [r["id"] for r in rows if not r.get("instructions", {}).get("zh")]
    print("缺少中文描述条数:", len(missing))
    print("缺少中文步骤条数:", len([r for r in rows if not r.get("instruction_steps", {}).get("zh")]))
    print("示例 id:", rows[0]["id"], rows[0]["name"])


if __name__ == "__main__":
    main()
