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
        name=constants.DEFAULT_COLLECTION_NAME
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
        'foo.py::TestClass.test_method (method)'
    ]
    source_code = ['\n'.join(code), '\n'.join(code[0:4]), '\n'.join(code[6:]), '\n'.join(code[7:])]
    metadatas = [
        {'type': 'file', 'id': 'foo.py', 'filepath': 'foo.py', 'text': source_code[0]},
        {'type': 'function', 'id': 'foo.py::test_func', 'filepath': 'foo.py', 'text': source_code[1]},
        {'type': 'class', 'id': 'foo.py::TestClass', 'filepath': 'foo.py', 'text': source_code[2]},
        {'type': 'method', 'id': 'foo.py::TestClass.test_method', 'filepath': 'foo.py', 'text': source_code[3]}
    ]

    collection.upsert(
        ids=ids,
        metadatas=metadatas,
        documents=['garbage'] * len(ids)
    )

    # do nothing
    llm_mock = lambda x: x

    tool = tools.CyclomaticComplexity('You are a tool.', collection, llm_mock)

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
