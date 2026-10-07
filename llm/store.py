from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from api.schema import AgentModelMap, AgentProfile, AgentProfileMap, ModelConfig, ModelPublic


class ModelStore:
    """模型配置仓库。

    配置以明文 JSON 保存，满足用户希望在前端直接新增、删除和测试模型的要求。
    文件默认被 `.gitignore` 忽略，避免 API key 被误提交。
    """

    def __init__(self, path: Path | None = None):
        # project_root 让默认配置路径不依赖启动命令当时的 cwd。
        project_root = Path(__file__).resolve().parents[1]
        # path 是模型配置文件路径；测试时可传入临时路径。
        self.path = path or project_root / "memory" / "config" / "models.json"
        # 确保配置目录存在，避免首次启动保存模型时报错。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 首次启动时创建空模型列表和默认智能体模型映射。
        if not self.path.exists():
            self._write(
                {
                    "models": [],
                    "agent_models": AgentModelMap().model_dump(),
                    "agent_profiles": AgentProfileMap().model_dump(),
                }
            )

    def all(self) -> list[ModelConfig]:
        """读取全部模型配置。"""

        # data 是 models.json 的完整内容。
        data = self._read()
        # 将原始 dict 转成 Pydantic 模型，保证字段类型和默认值一致。
        return [ModelConfig(**item) for item in data.get("models", [])]

    def redact_text(self, value: Any) -> str:
        """从任意可展示文本中移除已配置密钥和常见凭据字面量。"""

        text = str(value)
        # 真实配置值优先精确替换。过短测试值不参与，避免把普通字母误删。
        for model in self.all():
            secret = str(model.api_key or "")
            if len(secret) >= 8:
                text = text.replace(secret, "[密钥已脱敏]")
        # 即使模型凭空生成了一个未配置的密钥，也不能让它进入源码、事件或记忆。
        token_pattern = re.compile(r"(?i)\b(?:sk-|ak_|agt_)[A-Za-z0-9._-]{12,}\b")
        return token_pattern.sub("[密钥已脱敏]", text)

    def redact(self, value: Any) -> Any:
        """递归脱敏事件、结果和记忆载荷，同时保留原有结构。"""

        if isinstance(value, dict):
            return {str(key): self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact(item) for item in value)
        if isinstance(value, str):
            return self.redact_text(value)
        return value

    def contains_secret(self, value: Any) -> bool:
        """判断工具参数是否含真实或疑似凭据，供写入前的权限闸门使用。"""

        text = str(value)
        return self.redact_text(text) != text

    def model_map(self) -> AgentModelMap:
        """读取每个智能体绑定的模型 id。"""

        profiles = self.profiles()
        return AgentModelMap(
            **{name: str(profile["model_id"]) for name, profile in profiles.model_dump().items()}
        )

    def profiles(self) -> AgentProfileMap:
        """读取 Agent 的模型与思考等级，并兼容旧 agent_models 配置。"""

        data = self._read()
        raw = data.get("agent_profiles")
        if isinstance(raw, dict) and raw:
            return AgentProfileMap(**raw)
        legacy = data.get("agent_models") if isinstance(data.get("agent_models"), dict) else {}
        defaults = AgentProfileMap().model_dump()
        for name in defaults:
            defaults[name] = {
                "model_id": str(legacy.get(name) or ""),
                "effort": "inherit",
            }
        return AgentProfileMap(**defaults)

    def set_model_map(self, model_map: AgentModelMap) -> AgentModelMap:
        """保存智能体到模型 id 的映射。"""

        # data 是当前配置文件内容，只替换 agent_models 部分。
        data = self._read()
        # model_ids 是当前真实存在的模型；非空映射不能指向已删除或拼错的 id。
        model_ids = {str(item.get("id")) for item in data.get("models", [])}
        invalid = sorted({model_id for model_id in model_map.model_dump().values() if model_id and model_id not in model_ids})
        if invalid:
            raise ValueError("智能体模型映射引用了不存在的模型：" + "、".join(invalid))
        data["agent_models"] = model_map.model_dump()
        profiles = self.profiles().model_dump()
        for name, model_id in model_map.model_dump().items():
            profiles[name]["model_id"] = model_id
        data["agent_profiles"] = profiles
        self._write(data)
        return model_map

    def set_profiles(self, profiles: AgentProfileMap) -> AgentProfileMap:
        """保存 Agent 配置并同步旧版模型映射字段。"""

        data = self._read()
        model_ids = {str(item.get("id")) for item in data.get("models", [])}
        invalid = sorted(
            {
                profile.model_id
                for profile in profiles.__dict__.values()
                if isinstance(profile, AgentProfile)
                and profile.model_id
                and profile.model_id not in model_ids
            }
        )
        if invalid:
            raise ValueError("Agent 配置引用了不存在的模型：" + "、".join(invalid))
        data["agent_profiles"] = profiles.model_dump()
        data["agent_models"] = {
            name: item["model_id"] for name, item in profiles.model_dump().items()
        }
        self._write(data)
        return profiles

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
        profiles = self.profiles()
        profile = getattr(profiles, agent, AgentProfile())
        return self.get(profile.model_id or None)

    def effort_for_agent(self, agent: str, override: str | None = None) -> str:
        """返回 Agent 实际使用的思考等级。"""

        if override and override != "inherit":
            return override
        profile = getattr(self.profiles(), agent, AgentProfile())
        if profile.effort != "inherit":
            return profile.effort
        model = self.for_agent(agent)
        return model.default_effort

    def public(self) -> list[ModelPublic]:
        """返回不会泄露密钥的浏览器模型列表。"""

        result: list[ModelPublic] = []
        for model in self.all():
            key = model.api_key
            result.append(
                ModelPublic(
                    **model.model_dump(exclude={"api_key"}),
                    has_key=bool(key),
                    key_hint=(f"••••{key[-4:]}" if len(key) >= 4 else "已配置") if key else "",
                )
            )
        return result

    def fallbacks(self, current_id: str, excluded: set[str] | None = None) -> list[ModelConfig]:
        """返回当前模型失败后可用的启用模型，结构化任务把 LongCat 放到最后。"""

        blocked = set(excluded or set())
        blocked.add(current_id)
        # 保留配置文件顺序作为 fallback 优先级。用户可通过调整 models.json 顺序或 agent_models
        # 明确指定 LongCat 优先；这里不再为 JSON 任务强行把 LongCat 排到最后。
        return [model for model in self.all() if model.enabled and model.id not in blocked]

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
                value = cfg.model_dump()
                if not value.get("api_key"):
                    value["api_key"] = str(item.get("api_key") or "")
                models[idx] = value
                replaced = True
                break
        # 没有同 id 模型时追加为新配置。
        if not replaced:
            if not cfg.api_key:
                raise ValueError("新增模型必须填写 API Key")
            models.append(cfg.model_dump())
        data["models"] = models
        self._write(data)
        return cfg

    def copy_private(self, model_id: str, workdir: str, target: str = ".local/model.json") -> Path:
        """把模型配置复制到工作区内的私密文件，返回路径但不返回密钥。"""

        root = Path(workdir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("工作区不存在")
        destination = (root / target).resolve()
        if root not in destination.parents:
            raise ValueError("模型配置只能复制到当前工作区")
        if ".local" not in destination.relative_to(root).parts[:-1]:
            raise ValueError("模型配置只能复制到 .local 私密目录")
        ignore = root / ".gitignore"
        if ignore.is_symlink():
            raise ValueError("不能通过符号链接修改忽略配置")
        cfg = self.get(model_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        payload = {
            "id": cfg.id,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "api_key": cfg.api_key,
            "protocol": cfg.protocol,
            "timeout": cfg.timeout,
            "default_effort": cfg.default_effort,
        }
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        temp = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)
        os.chmod(destination, 0o600)
        ignore_text = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        entry = f"{destination.parent.relative_to(root).as_posix().rstrip('/')}/"
        if entry not in {line.strip() for line in ignore_text.splitlines()}:
            with ignore.open("a", encoding="utf-8") as fh:
                if ignore_text and not ignore_text.endswith("\n"):
                    fh.write("\n")
                fh.write(entry + "\n")
        return destination

    def delete(self, model_id: str) -> None:
        """删除模型，并把引用该模型的智能体映射切换到 fallback。"""

        # data 是当前配置文件内容。
        data = self._read()
        # 删除指定 id 的模型，保留其他模型。
        data["models"] = [item for item in data.get("models", []) if item.get("id") != model_id]
        # model_map 是删除前的智能体模型映射。
        model_map = AgentModelMap(**data.get("agent_models", {}))
        # fallback 是删除后可用的第一个模型 id；没有模型时为空字符串。
        enabled = [item for item in data["models"] if item.get("enabled", True)]
        fallback = (enabled or data["models"])[0]["id"] if data["models"] else ""
        # mapped 是可修改的映射 dict。
        mapped = model_map.model_dump()
        # 如果某个智能体原来绑定被删除模型，就切换到 fallback，避免前端下拉框悬空。
        for agent, current in list(mapped.items()):
            if current == model_id:
                mapped[agent] = fallback
        data["agent_models"] = mapped
        profiles = self.profiles().model_dump()
        for profile in profiles.values():
            if profile["model_id"] == model_id:
                profile["model_id"] = fallback
        data["agent_profiles"] = profiles
        self._write(data)

    def _read(self) -> dict[str, Any]:
        """读取原始 JSON 配置。"""

        with self.path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, data: dict[str, Any]) -> None:
        """把配置写回 JSON 文件。"""

        # content 是格式化后的完整配置，便于用户直接检查明文文件。
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        # temp_path 与正式配置同目录，写完后原子替换，避免程序中断留下半个 JSON。
        temp_path = self.path.with_name(f".{self.path.name}.tmp")
        temp_path.write_text(content, encoding="utf-8")
        os.chmod(temp_path, 0o600)
        temp_path.replace(self.path)
        os.chmod(self.path, 0o600)
