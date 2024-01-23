"""
Tests for tools.
"""
import chromadb

from groundcrew import constants
from groundcrew import tools
from groundcrew.tools import cyclomatic_complexity


# def test_cyclomatic_complexity():
#     code = (
#         'def foo(x):\n'
#         '    if x > 0:\n'
#         '        return x\n'
#         '    else:\n'
#         '        return -x\n'
#     )
#     cc = cyclomatic_complexity(code)
#     assert cc['foo']['object'] == 'function'
#     assert cc['foo']['complexity'] == 2

#     code = (
#         'class Foo:\n'
#         '    def __init__(self):\n'
#         '        self.x = 0\n'
#         '    def bar(self):\n'
#         '        if self.x > 0:\n'
#         '            return self.x\n'
#         '        else:\n'
#         '            return -self.x\n'
#     )
#     cc = cyclomatic_complexity(code)
#     assert cc['Foo']['object'] == 'class'
#     assert cc['Foo']['complexity'] == 3


def test_complexity_tool():
    """tests for complexitytool"""

    # in memory client for testing
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name='test'
    )

    code = (
        'def test_func(x):',
        '    if x > 0:',
        '        return x',
        '    else:',
        '        return -x',
        '',
        'class TestClass:',
        '    def test_method(self):',
        '        if self.x > 0:',
        '            return self.x',
        '        elif self.x == 0:',
        '            return 0',
        '        else:',
        '            return -self.x'
    )
    ids = [
        'foo.py',
        'foo.py::test_func (function)',
        'foo.py::TestClass (class)',
        'foo.py::TestClass.test_method (method)',
        'bar.py',
        'bar.py::test_func (function)',
        'README.md'
    ]
    source_code = [
        '\n'.join(code),
        '\n'.join(code[0:5]),
        '\n'.join(code[6:]),
        '\n'.join(code[7:]),
        '\n'.join(code[0:6]),
        '\n'.join(code[0:5]),
        'read me'
    ]
    metadatas = [
        {'type': 'file', 'id': 'foo.py', 'filepath': 'foo.py', 'text': source_code[0]},
        {'type': 'function', 'id': 'foo.py::test_func', 'filepath': 'foo.py', 'text': source_code[1]},
        {'type': 'class', 'id': 'foo.py::TestClass', 'filepath': 'foo.py', 'text': source_code[2]},
        {'type': 'method', 'id': 'foo.py::TestClass.test_method', 'filepath': 'foo.py', 'text': source_code[3]},
        {'type': 'file', 'id': 'bar.py', 'filepath': 'bar.py', 'text': source_code[4]},
        {'type': 'function', 'id': 'bar.py::test_func', 'filepath': 'bar.py', 'text': source_code[5]},
        {'type': 'file', 'id': 'README.md', 'filepath': 'README.md', 'text': source_code[6]},
    ]

    collection.upsert(
        ids=ids,
        metadatas=metadatas,
        documents=['garbage'] * len(ids)
    )

    # do nothing
    llm_mock = lambda x: x

    tool = tools.CyclomaticComplexityTool('You are a tool.', collection, llm_mock)
    assert set(tool.get_files().keys()) == {'foo.py', 'bar.py'}
    assert tool.get_complexity(source_code[0]) == {
        'average': 3.0,
        'max': 4,
        'details': {
            'TestClass': {
                'complexity': 4,
                'methods': {'test_method': {'complexity': 3}},
                'object': 'class'
            },
            'test_func': {'complexity': 2, 'object': 'function'}
        }
    }
    output = tool.complexity_analysis('max')
    target_output = [
        'Cyclomatic complexity summary for the most complex files:',
        'File: foo.py; average complexity = 3.0; max complexity = 4',
        '  test_func: 2',
        '  TestClass: 4',
        '    TestClass.test_method: 3',
        'File: bar.py; average complexity = 2.0; max complexity = 2',
        '  test_func: 2',
        ''
    ]
    assert output.split('\n') == target_output

    output = tool.complexity_analysis('average')
    assert output.split('\n') == target_output


def test_lintfiletool():
    """tests for lintfiletool"""

    # in memory client for testing
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=constants.DEFAULT_COLLECTION_NAME
    )

    example_ids_metadatas = [
        (str(idx), dict(filepath=f'src/{name}.py'))
        for idx, name in enumerate(['apples', 'bananas', 'oranges'])
    ]

    collection.upsert(
        ids=[x for x, _ in example_ids_metadatas],
        metadatas=[y for _, y in example_ids_metadatas],
        documents=['garbage'] * len(example_ids_metadatas)
    )

    # do nothing
    llm_mock = lambda x: x

    tool = tools.LintFileTool('You are a tool.', collection, llm_mock, 'derp')
    tool.working_dir_path = 'baloney'

    # test fuzzy matching
    assert tool.fuzzy_match_file_path('bapples', 50) == 'src/apples.py'    # close
    assert tool.fuzzy_match_file_path('oranges', 50) == 'src/oranges.py'   # exact
    assert tool.fuzzy_match_file_path('baloney', 50) == 'src/bananas.py'   # far
    assert tool.fuzzy_match_file_path('baloney', 75) is None

    # test getting filepaths from collection
    assert tool.get_paths() == {
        '0': 'src/apples.py',
        '1': 'src/bananas.py',
        '2': 'src/oranges.py'
    }
