"""结构化 fixture 事实帮助审查发现局部替身覆盖与遗漏依赖。"""

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
