"""技能读取、工具参数与结果在恢复后保持完整且脱敏。"""
from api.schema import ModelConfig, SkillContext
from backend.agents.graph import AgentGraph
from backend.agents.manager import manager
from backend.memory import MemoryStore
from llm.store import ModelStore


def test_skill_and_tool_details_survive_history(tmp_path):
    models = ModelStore(tmp_path / 'models.json')
    models.upsert(ModelConfig(id='test', name='测试', model='test', base_url='https://example.test', api_key='TEST_PRIVATE_VALUE'))
    memory = MemoryStore(tmp_path / 'mem')
    graph = AgentGraph(models, memory)
    state = {'session_id': 'details', 'workdir': str(tmp_path / 'project'), 'text': '规划一个页面', 'plan_mode': True,
             'skills': [SkillContext(id='design', name='设计', description='布局说明', content='布局内容 TEST_PRIVATE_VALUE')]}
    manager(graph, state)
    graph._emit(state, 'coder', 'tool', '读取完成', data={'arguments': {'path': 'main.py'}, 'result': {'text': '完整返回 TEST_PRIVATE_VALUE'}})
    restored = memory.list_history()[0]['sessions'][0]['events']
    skill = next(event for event in restored if event['kind'] == 'skill')
    assert skill['data']['读取内容'] == '布局内容 [密钥已脱敏]'
    tool = restored[-1]['data']
    assert tool['arguments']['path'] == 'main.py'
    assert tool['result']['text'] == '完整返回 [密钥已脱敏]'
    assert 'TEST_PRIVATE_VALUE' not in str(restored)
