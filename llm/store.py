# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

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
        """初始化当前对象及其运行状态。"""
        project_root = Path(__file__).resolve().parents[1]
        self.path = path or project_root / "memory" / "config" / "models.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
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

        data = self._read()
        return [ModelConfig(**item) for item in data.get("models", [])]

    def redact_text(self, value: Any) -> str:
        """从任意可展示文本中移除已配置密钥和常见凭据字面量。"""

        text = str(value)
        for model in self.all():
            secret = str(model.api_key or "")
            if len(secret) >= 8:
                text = text.replace(secret, "[密钥已脱敏]")
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

    def secret_facts(self, value: Any) -> dict[str, int]:
        """只返回拦截类别计数，便于诊断且不保存任何凭据或原始参数。"""

        text = str(value)
        return {
            "configured_matches": sum(len(model.api_key) >= 8 and model.api_key in text for model in self.all()),
            "token_like_matches": len(re.findall(r"(?i)\b(?:sk-|ak_|agt_)[A-Za-z0-9._-]{12,}\b", text)),
        }

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

        data = self._read()
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
        """获取某个智能体应该使用的模型配置。"""

        if override:
            return self.get(override)
        profiles = self.profiles()
        profile = getattr(profiles, agent, AgentProfile())
        return self.get(profile.model_id or None)

    def effort_for_agent(self, agent: str, override: str | None = None, *, model: ModelConfig | None = None) -> str:
        """继承实际选中模型的默认等级，换模和视觉路由不再查回原模型。"""

        if override and override != "inherit":
            chosen = override
        else:
            chosen = getattr(self.profiles(), agent, AgentProfile()).effort
        if model is None:
            if chosen != "inherit":
                return chosen
            model = self.for_agent(agent)
        return chosen if chosen in model.efforts else model.default_effort

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
        """按配置顺序返回当前模型之外的启用备用模型。"""

        blocked = set(excluded or set())
        blocked.add(current_id)
        return [model for model in self.all() if model.enabled and model.id not in blocked]

    def upsert(self, cfg: ModelConfig) -> ModelConfig:
        """新增或更新一个模型配置。"""

        data = self._read()
        models = data.get("models", [])
        replaced = False
        for idx, item in enumerate(models):
            if item.get("id") == cfg.id:
                value = cfg.model_dump()
                if not value.get("api_key"):
                    value["api_key"] = str(item.get("api_key") or "")
                models[idx] = value
                replaced = True
                break
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
        relative_parts = destination.relative_to(root).parts
        project_root = root.joinpath(*relative_parts[:relative_parts.index(".local")])
        ignore = project_root / ".gitignore"
        if ignore.is_symlink():
            raise ValueError("不能通过符号链接修改忽略配置")
        configured = self.all()
        cfg = next((item for item in configured if item.id == model_id), None)
        if cfg is None:
            matches = [item for item in configured if item.enabled and model_id in {item.model, item.name}]
            if len(matches) != 1:
                raise ValueError("模型名称未唯一匹配配置，请使用配置 ID")
            cfg = matches[0]
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
            "max_output_tokens": cfg.max_output_tokens,
        }
        from llm.client import LlmClient

        client = LlmClient(cfg)
        payload["request_options"] = client._chat_reasoning(cfg.default_effort)
        payload["output_token_field"] = client.chat_token_field()
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
        entry = f"{destination.parent.relative_to(project_root).as_posix().rstrip('/')}/"
        if entry not in {line.strip() for line in ignore_text.splitlines()}:
            with ignore.open("a", encoding="utf-8") as fh:
                if ignore_text and not ignore_text.endswith("\n"):
                    fh.write("\n")
                fh.write(entry + "\n")
        return destination

    def delete(self, model_id: str) -> None:
        """删除模型，并把引用该模型的智能体映射切换到 fallback。"""

        data = self._read()
        data["models"] = [item for item in data.get("models", []) if item.get("id") != model_id]
        model_map = AgentModelMap(**data.get("agent_models", {}))
        enabled = [item for item in data["models"] if item.get("enabled", True)]
        fallback = (enabled or data["models"])[0]["id"] if data["models"] else ""
        mapped = model_map.model_dump()
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

        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        temp_path = self.path.with_name(f".{self.path.name}.tmp")
        temp_path.write_text(content, encoding="utf-8")
        os.chmod(temp_path, 0o600)
        temp_path.replace(self.path)
        os.chmod(self.path, 0o600)
