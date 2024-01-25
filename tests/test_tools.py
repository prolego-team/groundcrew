"""
Tests for tools.
"""
import chromadb

from groundcrew import constants, tools


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
    assert out is True

    out = tool._id_matches_file(
        id_='src/apples.py',
        filename='src/apples.py',
        function_name='none')
    assert out is True

    out = tool._id_matches_file(
        id_='apples.py',
        filename='apples.py',
        function_name='foo')
    assert out is False

    out = tool._id_matches_file(
        id_='apples.py::foo (function)',
        filename='apples.py',
        function_name='foo')
    assert out is True

    out = tool._id_matches_file(
        id_='apples.py::foo (function)',
        filename='apples.py',
        function_name='none')
    assert out is True

    out = tool._id_matches_file(
        id_='apples.py::foo (function)',
        filename='bananas.py',
        function_name='none')
    assert out is False

    #### End test _id_matches_file ####

    #### Test _id_matches_function ####

    out = tool._id_matches_function(
        id_='apples.py::foo (function)',
        function_name='foo')
    assert out is True

    out = tool._id_matches_function(
        id_='apples.py::foo (function)',
        function_name='none')
    assert out is False

    out = tool._id_matches_function(
        id_='bananas.py',
        function_name='foo')
    assert out is False

    out = tool._id_matches_function(
        id_='bananas.py',
        function_name='none')
    assert out is False

    #### End test _id_matches_function ####


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
