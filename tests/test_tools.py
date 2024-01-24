"""
Tests for tools.
"""
import chromadb

from groundcrew import constants
from groundcrew import tools


def test_findusage_tool():
    """tests for findusagetool"""

    # in memory client for testing
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name='test'
    )

    ids = [
        'foo.py',
        'bar.py',
        'README.md'
    ]
    source_code = [
        (
            'import numpy as np\n'
            'from xyz import abc\n'
            'print(np.random.rand())\n'
            '# more code\n'
            'print(np.dot(x,y))\n'
            'abc(1), abc(2)'
        ),
        (
            'from torch.nn import Module, Linear\n'
            'import numpy as np\n'
            'from xyz import abc\n'
            'print(Module())\n'
            'print(Module())\n'
            'abc(1)'
        ),
        'read me'
    ]
    metadatas = [
        {'type': 'file', 'id': 'foo.py', 'filepath': 'foo.py', 'text': source_code[0]},
        {'type': 'file', 'id': 'bar.py', 'filepath': 'bar.py', 'text': source_code[1]},
        {'type': 'file', 'id': 'README.md', 'filepath': 'README.md', 'text': source_code[2]},
    ]

    collection.upsert(
        ids=ids,
        metadatas=metadatas,
        documents=['garbage'] * len(ids)
    )

    llm_mock = lambda x: x

    tool = tools.FindUsageTool('You are a tool.', collection, llm_mock)
    assert tool.get_usage('numpy') == {'foo.py': 2}
    assert tool.get_usage('numpy.rand') == {'foo.py': 1}
    assert tool.get_usage('numpy.random.rand') == {'foo.py': 1}
    assert tool.get_usage('numpy.random.rand') == {'foo.py': 1}
    assert tool.get_usage('torch.nn.Module') == {'bar.py': 2}
    assert tool.get_usage('torch.nn.Linear') == {}
    assert tool.get_usage('xyz.abc') == {'foo.py':2, 'bar.py': 1}

    assert tool.summarize_usage({'foo.py': 2, 'bar.py': 1}) == (
        'Usage summary (filename: usage count):\n'
        'foo.py: 2\n'
        'bar.py: 1\n'
    )
    assert tool.summarize_usage({'test.py': 999}) == (
        'Usage summary (filename: usage count):\n'
        'test.py: 999\n'
    )
    assert tool.summarize_usage({}) == (
        'Usage summary (filename: usage count):\n'
        'No usage found.\n'
    )


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
    assert set(tools.get_python_files(collection).keys()) == {'foo.py', 'bar.py'}
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
