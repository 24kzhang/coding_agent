from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api.schema import AgentModelMap, ModelConfig


class ModelStore:
    """模型配置仓库。

    配置以明文 JSON 保存，满足用户希望在前端直接新增、删除和测试模型的要求。
    文件默认被 `.gitignore` 忽略，避免 API key 被误提交。
    """

    def __init__(self, path: Path | None = None):
        self.path = path or Path("memory/config/models.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"models": [], "agent_models": AgentModelMap().model_dump()})

    def all(self) -> list[ModelConfig]:
        data = self._read()
        return [ModelConfig(**item) for item in data.get("models", [])]

    def model_map(self) -> AgentModelMap:
        data = self._read()
        return AgentModelMap(**data.get("agent_models", {}))

    def set_model_map(self, model_map: AgentModelMap) -> AgentModelMap:
        data = self._read()
        data["agent_models"] = model_map.model_dump()
        self._write(data)
        return model_map

    def get(self, model_id: str | None) -> ModelConfig:
        models = self.all()
        if not models:
            raise KeyError("还没有配置任何模型")
        if model_id:
            for model in models:
                if model.id == model_id:
                    return model
            raise KeyError(f"模型不存在：{model_id}")
        for model in models:
            if model.enabled:
                return model
        return models[0]

    def for_agent(self, agent: str, override: str | None = None) -> ModelConfig:
        if override:
            return self.get(override)
        model_map = self.model_map()
        return self.get(getattr(model_map, agent, None))

    def upsert(self, cfg: ModelConfig) -> ModelConfig:
        data = self._read()
        models = data.get("models", [])
        replaced = False
        for idx, item in enumerate(models):
            if item.get("id") == cfg.id:
                models[idx] = cfg.model_dump()
                replaced = True
                break
        if not replaced:
            models.append(cfg.model_dump())
        data["models"] = models
        self._write(data)
        return cfg

    def delete(self, model_id: str) -> None:
        data = self._read()
        data["models"] = [item for item in data.get("models", []) if item.get("id") != model_id]
        model_map = AgentModelMap(**data.get("agent_models", {}))
        fallback = data["models"][0]["id"] if data["models"] else ""
        mapped = model_map.model_dump()
        for agent, current in list(mapped.items()):
            if current == model_id:
                mapped[agent] = fallback
        data["agent_models"] = mapped
        self._write(data)

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, data: dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
