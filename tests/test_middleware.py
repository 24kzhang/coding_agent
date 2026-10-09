"""HTTP 中间件与路由的异常处理边界不能混淆。"""

from backend.agents.precheck import middleware_issues
from backend.tools import FsTool


def test_uncaught_route_exception_in_middleware_is_reported(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("main.py", 'from fastapi import HTTPException as RouteError\n@app.middleware("http")\nasync def guard(request, call_next):\n    raise RouteError(403, "禁止访问")\n')
    issues = middleware_issues(fs, {}, ["main.py"])
    assert len(issues) == 1 and "main.py:4" in issues[0]


def test_routes_caught_exceptions_and_shadowed_names_are_not_reported(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("main.py", '''from fastapi import HTTPException
@app.get("/items")
async def endpoint():
    raise HTTPException(404)
@app.middleware("http")
async def caught(request, call_next):
    try:
        raise HTTPException(403)
    except HTTPException:
        return Response(status_code=403)
@app.middleware("http")
async def shadowed(request, call_next, HTTPException):
    raise HTTPException(403)
''')
    assert middleware_issues(fs, {}, ["main.py"]) == []
