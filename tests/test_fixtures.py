"""结构化 fixture 事实帮助审查发现局部替身覆盖与遗漏依赖。"""

from backend.agents.precheck import import_issues
from backend.agents.verifier import fixture_contracts
from backend.tools import FsTool


def test_fixture_contracts_distinguish_explicit_dependencies_and_autouse(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("tests/conftest.py", '''import pytest
@pytest.fixture()
def fake_sender(monkeypatch):
    monkeypatch.setattr(Service, "send", fake)
@pytest.fixture(autouse=True)
def guard_network(monkeypatch):
    monkeypatch.setenv("OFFLINE", "1")
@pytest.fixture()
def client():
    return Client()
''')
    fs.write("tests/test_api.py", "def test_mocked(client, fake_sender):\n    client.post('/')\ndef test_uncovered(client):\n    client.post('/')\n")
    facts = fixture_contracts(fs, ["tests/conftest.py", "tests/test_api.py"])
    by_name = {fact["名称"]: fact for fact in facts}
    assert by_name["fake_sender"]["自动应用"] is False
    assert by_name["guard_network"]["自动应用"] is True
    assert by_name["client"]["依赖参数"] == []
    assert "fake_sender" in by_name["test_mocked"]["依赖参数"]
    assert "fake_sender" not in by_name["test_uncovered"]["依赖参数"]
    assert 'Service, \'send\'' in by_name["fake_sender"]["补丁目标"][0]


def test_mock_contract_exposes_original_global_state_commit(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("service.py", "registry = {}\ndef deliver(item):\n    reply = remote_send(item)\n    registry[item.id] = item\n    return reply\n")
    fs.write("test_api.py", "import service as mod\ndef test_flow(monkeypatch):\n    monkeypatch.setattr(mod, 'deliver', lambda item: '回复')\n")
    facts = []
    assert not import_issues(fs, {}, ["service.py", "test_api.py"], contracts=facts)
    assert facts[0]["原函数的模块状态写入"] == ["registry[item.id] = item"]
    assert facts[0]["替身表达式"] == "lambda item: '回复'"
    fs.write("service.py", "registry = {}\ndef deliver(item, registry):\n    registry[item.id] = item\n    return '回复'\n")
    facts = []
    import_issues(fs, {}, ["test_api.py"], contracts=facts)
    assert not facts
