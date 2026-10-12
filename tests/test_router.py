"""工厂依赖隔离的路由预检，不限制正常模块级单例或幂等注册。"""

import pytest

from backend.agents.precheck import router_issues
from backend.tools import FsTool


@pytest.mark.parametrize("setup,expected", [
    ("", True),
    ("    router = APIRouter()\n", False),
    ("    router.routes.clear()\n", False),
])
def test_factory_does_not_accumulate_global_route_closures(tmp_path, setup, expected):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("routes.py", "from fastapi import APIRouter\nrouter=APIRouter()\ndef build_routes(service):\n" + setup + "    @router.get('/items')\n    def items():\n        return service.items()\n    return router\n")
    assert bool(router_issues(fs, {}, ["routes.py"])) is expected


def test_module_level_singleton_routes_remain_valid(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("routes.py", "from fastapi import APIRouter\nrouter=APIRouter()\n@router.get('/items')\ndef items():\n    return []\n")
    assert not router_issues(fs, {}, ["routes.py"])


def test_app_factory_must_not_mount_reused_closure_router(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    source = "from fastapi import APIRouter, FastAPI\nrouter=APIRouter()\ndef create_app(config):\n    app=FastAPI()\n    @router.get('/config')\n    def current_config():\n        return config\n    app.include_router(router)\n    return app\n"
    fs.write("main.py", source)
    assert router_issues(fs, {}, ["main.py"])
    fs.write("main.py", source.replace("    app=FastAPI()", "    router=APIRouter()\n    app=FastAPI()"))
    assert not router_issues(fs, {}, ["main.py"])
