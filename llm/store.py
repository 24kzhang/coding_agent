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
        # path 是模型配置文件路径；测试时可传入临时路径，生产默认写 memory/config/models.json。
        self.path = path or Path("memory/config/models.json")
        # 确保配置目录存在，避免首次启动保存模型时报错。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 首次启动时创建空模型列表和默认智能体模型映射。
        if not self.path.exists():
            self._write({"models": [], "agent_models": AgentModelMap().model_dump()})

    def all(self) -> list[ModelConfig]:
        """读取全部模型配置。"""

        # data 是 models.json 的完整内容。
        data = self._read()
        # 将原始 dict 转成 Pydantic 模型，保证字段类型和默认值一致。
        return [ModelConfig(**item) for item in data.get("models", [])]

    def model_map(self) -> AgentModelMap:
        """读取每个智能体绑定的模型 id。"""

        # data 是 models.json 的完整内容。
        data = self._read()
        return AgentModelMap(**data.get("agent_models", {}))

    def set_model_map(self, model_map: AgentModelMap) -> AgentModelMap:
        """保存智能体到模型 id 的映射。"""

        # data 是当前配置文件内容，只替换 agent_models 部分。
        data = self._read()
        data["agent_models"] = model_map.model_dump()
        self._write(data)
        return model_map

    def get(self, model_id: str | None) -> ModelConfig:
        """按模型 id 获取配置；为空时返回第一个启用模型。"""

        # models 是当前已配置模型列表。
        models = self.all()
        if not models:
            raise KeyError("还没有配置任何模型")
        if model_id:
            # 显式传入 model_id 时必须精确匹配，否则提示配置错误。
            for model in models:
                if model.id == model_id:
                    return model
            raise KeyError(f"模型不存在：{model_id}")
        # 没有显式指定模型时，优先选择 enabled=True 的模型。
        for model in models:
            if model.enabled:
                return model
        # 如果所有模型都被禁用，仍返回第一个，避免系统完全不可用。
        return models[0]

    def for_agent(self, agent: str, override: str | None = None) -> ModelConfig:
        """获取某个智能体应该使用的模型配置。"""

        # override 是本次请求的临时模型覆盖，优先级高于全局 agent_models。
        if override:
            return self.get(override)
        # model_map 保存每个智能体对应的模型 id。
        model_map = self.model_map()
        return self.get(getattr(model_map, agent, None))

    def upsert(self, cfg: ModelConfig) -> ModelConfig:
        """新增或更新一个模型配置。"""

        # data 是当前配置文件内容。
        data = self._read()
        # models 是原始模型 dict 列表，写回时仍保持 JSON 结构。
        models = data.get("models", [])
        # replaced 标记当前 id 是否已经存在。
        replaced = False
        # idx 是当前遍历到的模型下标，item 是模型原始 dict。
        for idx, item in enumerate(models):
            if item.get("id") == cfg.id:
                models[idx] = cfg.model_dump()
                replaced = True
                break
        # 没有同 id 模型时追加为新配置。
        if not replaced:
            models.append(cfg.model_dump())
        data["models"] = models
        self._write(data)
        return cfg

    def delete(self, model_id: str) -> None:
        """删除模型，并把引用该模型的智能体映射切换到 fallback。"""

        # data 是当前配置文件内容。
        data = self._read()
        # 删除指定 id 的模型，保留其他模型。
        data["models"] = [item for item in data.get("models", []) if item.get("id") != model_id]
        # model_map 是删除前的智能体模型映射。
        model_map = AgentModelMap(**data.get("agent_models", {}))
        # fallback 是删除后可用的第一个模型 id；没有模型时为空字符串。
        fallback = data["models"][0]["id"] if data["models"] else ""
        # mapped 是可修改的映射 dict。
        mapped = model_map.model_dump()
        # 如果某个智能体原来绑定被删除模型，就切换到 fallback，避免前端下拉框悬空。
        for agent, current in list(mapped.items()):
            if current == model_id:
                mapped[agent] = fallback
        data["agent_models"] = mapped
        self._write(data)

    def _read(self) -> dict[str, Any]:
        """读取原始 JSON 配置。"""

        with self.path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, data: dict[str, Any]) -> None:
        """把配置写回 JSON 文件。"""

        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
