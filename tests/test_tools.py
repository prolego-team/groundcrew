"""
Tests for tools.
"""
import chromadb

from groundcrew import constants, tools


def _clear_collection(collection: chromadb.Collection) -> None:
    """clear a collection"""
    uids = collection.get()['ids']
    if uids:
        collection.delete(ids=uids)


def test_singledocstringtool():
    """
    Tests for SingleDocstringTool
    """

    llm_mock = lambda x: x

    # In memory client for testing
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=constants.DEFAULT_COLLECTION_NAME
    )
    _clear_collection(collection)

    example_ids = [
        'apples.py',
        'apples.py::foo (function)',
        'bananas.py',
    ]
    example_metadatas = [
        dict(filepath=f'src/{name}.py', text=f'{name} code')
        for name in example_ids
    ]
    collection.upsert(
        ids=example_ids,
        metadatas=example_metadatas,
        documents=['document'] * len(example_ids)
    )

    all_ids = collection.get()['ids']

    tool = tools.SingleDocstringTool('You are a tool.', collection, llm_mock)

    #### Test _determine_context ####

    # All are none - model didn't pass anything in
    context = tool._determine_context(
        code='none', filename='none', function_name='none')
    assert context == 'no_match'

    # Model only passed in code
    code = 'def foo():\n    """\n    This is a docstring.\n    """\n    pass'
    context = tool._determine_context(
        code, filename='none', function_name='none')
    assert context == 'code'

    # Model passed in code and a function name - function name should be ignored
    code = 'def foo():\n    """\n    This is a docstring.\n    """\n    pass'
    context = tool._determine_context(
        code, filename='none', function_name='foo')
    assert context == 'code'

    # Model passed in a filename and a function name
    context = tool._determine_context(
        code='none', filename='functions.py', function_name='foo')
    assert context == 'file-function'

    #### End test _determine_context ####

    #### Test _get_function_code ####

    # Model passed in a filename and no function name - return all code from
    # the matched file
    function_code = tool._get_function_code(
        all_ids=all_ids,
        filename='apples.py',
        function_name='none',
        code='none',
        context='file'
    )
    expected_function_code = 'apples.py code\n\napples.py::foo (function) code\n'
    assert function_code == expected_function_code

    # Model only passed in code - generate docstring for just that code
    code = 'def foo():\n    """\n    This is a docstring.\n    """\n    pass'
    function_code = tool._get_function_code(
        all_ids=all_ids,
        filename='none',
        function_name='none',
        code=code,
        context='code'
    )
    assert function_code == code

    # Model passed in a function name - generate docstring for just the matched
    # function
    function_code = tool._get_function_code(
        all_ids=all_ids,
        filename='none',
        function_name='foo',
        code='none',
        context='function'
    )
    expected_function_code = 'apples.py::foo (function) code\n'
    assert function_code == expected_function_code

    # Model passed in a filename and a function name - generate docstring for
    # that specific function in that file
    function_code = tool._get_function_code(
        all_ids=all_ids,
        filename='apples.py',
        function_name='foo',
        code='none',
        context='function'
    )
    expected_function_code = 'apples.py::foo (function) code\n'
    assert function_code == expected_function_code

    #### End test _get_function_code ####

    #### Test _id_matches_file ####

    out = tool._id_matches_file(
        id_='apples.py',
        filename='apples.py',
        function_name='none')
    assert out

    out = tool._id_matches_file(
        id_='src/apples.py',
        filename='src/apples.py',
        function_name='none')
    assert out

    out = tool._id_matches_file(
        id_='apples.py',
        filename='apples.py',
        function_name='foo')
    assert not out

    out = tool._id_matches_file(
        id_='apples.py::foo (function)',
        filename='apples.py',
        function_name='foo')
    assert out

    out = tool._id_matches_file(
        id_='apples.py::foo (function)',
        filename='apples.py',
        function_name='none')
    assert out

    out = tool._id_matches_file(
        id_='apples.py::foo (function)',
        filename='bananas.py',
        function_name='none')
    assert not out

    #### End test _id_matches_file ####

    #### Test _id_matches_function ####

    out = tool._id_matches_function(
        id_='apples.py::foo (function)',
        function_name='foo')
    assert out

    out = tool._id_matches_function(
        id_='apples.py::foo (function)',
        function_name='none')
    assert not out

    out = tool._id_matches_function(
        id_='bananas.py',
        function_name='foo')
    assert not out

    out = tool._id_matches_function(
        id_='bananas.py',
        function_name='none')
    assert not out

    #### End test _id_matches_function ####


def test_get_python_files():
    """tests for findusagetool"""

    # in memory client for testing
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name='test_python_files'
    )

    metadatas = [
        {'type': 'file', 'id': 'foo.py', 'filepath': 'foo.py', 'text': 'File 1'},
        {'type': 'file', 'id': 'bar.py', 'filepath': 'bar.py', 'text': 'File 2'},
        {'type': 'file', 'id': 'README.md', 'filepath': 'README.md', 'text': 'File 3'},
        {'type': 'file', 'id': 'test.pyc', 'filepath': 'test.pyc', 'text': 'File 4'},
        {'type': 'file', 'id': 'baz.py', 'filepath': 'baz.py', 'text': 'File 5'},
    ]
    ids = [metadata['id'] for metadata in metadatas]

    collection.upsert(
        ids=ids,
        metadatas=metadatas,
        documents=['fake file summaries'] * len(ids)
    )

    assert tools.get_python_files(collection) == {
        'foo.py': 'File 1',
        'bar.py': 'File 2',
        'baz.py': 'File 5'
    }


def test_findusage_tool():
    """tests for findusagetool"""

    # in memory client for testing
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name='test_findusage_tool'
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
        'foo.py: 2\n'
        'bar.py: 1\n'
    )
    assert tool.summarize_usage({'test.py': 999}) == (
        'test.py: 999\n'
    )
    assert tool.summarize_usage({}) == (
        'No usage found.\n'
    )


def test_complexity_tool():
    """tests for complexitytool"""

    # in memory client for testing
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name='test_complexity_tool'
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
    all_files = tools.get_python_files(collection)

    # do nothing
    llm_mock = lambda x: x

    tool = tools.CyclomaticComplexityTool('You are a tool.', collection, llm_mock)
    assert set(tools.get_python_files(collection).keys()) == {'foo.py', 'bar.py'}
    assert tool._CyclomaticComplexityTool__get_complexity(source_code[0]) == {
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
    output = tool.complexity_analysis(files=all_files, sort_on='max')
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

    output = tool.complexity_analysis(files=all_files, sort_on='average')
    assert output.split('\n') == target_output

    output = tool.complexity_analysis(files={'foo.py': all_files['foo.py']}, sort_on='average')
    assert output.split('\n') == target_output[:5] + ['']


def test_lintfiletool():
    """tests for lintfiletool"""

    # in memory client for testing
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name='test_lintfiletool'
    )
    _clear_collection(collection)

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
    assert tools.fuzzy_match_file_path(collection, 'bapples', 50) == 'src/apples.py'    # close
    assert tools.fuzzy_match_file_path(collection, 'oranges', 50) == 'src/oranges.py'   # exact
    assert tools.fuzzy_match_file_path(collection, 'baloney', 50) == 'src/bananas.py'   # far
    assert tools.fuzzy_match_file_path(collection, 'baloney', 75) is None

    # test getting filepaths from collection
    assert tools.get_paths(collection) == {
        '0': 'src/apples.py',
        '1': 'src/bananas.py',
        '2': 'src/oranges.py'
    }

