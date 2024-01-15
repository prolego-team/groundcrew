"""
"""
import os
import ast

from git import Repo

from groundcrew.constants import DEFAULT_EF
from groundcrew.dataclasses import Chunk
from groundcrew.code_utils import get_imports_from_code, import_called_as

opj = os.path.join


def format_chunk(chunk: Chunk, include_text: bool) -> str:

    # TODO - change Document based on chunk type
    # TODO - start and end lines won't be needed if typ is a file

    prompt = f'Name: {chunk.uid}\n'
    prompt += f'Type: {chunk.typ}\n'
    if include_text:
        prompt += f'Full Text: {chunk.text}\n'
    prompt += f'Document: {chunk.document}\n'
    prompt += f'Start Line: {chunk.start_line}\n'
    prompt += f'End Line: {chunk.end_line}\n'
    return prompt


def get_committed_files(
        repository_dir: str, extensions: list[str]) -> list[str]:
    """
    This function finds all files in a directory and filters out those not
    committed to the repository

    Args:
        repository_dir: str
        extensions: list[str]

    Returns:
        list[str]
    """

    committed_files = set()
    repo = Repo(repository_dir)
    for blob in repo.head.commit.tree.traverse():
        file_path = os.path.join(repo.working_dir, blob.path)
        ext = os.path.splitext(file_path)[1]
        if ext in extensions:
            committed_files.add(file_path)

    return committed_files


def extract_python_from_file(file_text, node_type):

    file_lines = file_text.split('\n')
    texts = {}
    for node in ast.walk(ast.parse(file_text)):
        if isinstance(node, node_type):
            texts[node.name] = {
                'text': ''.join(
                    file_lines[node.lineno - 1:node.end_lineno]
                ),
                'start_line': node.lineno,
                'end_line': node.end_lineno
            }
    return texts


def generate_function_descriptions(
        llm: Callable[[str], str],
        function_descriptions: Dict[str, str],
        repository: str,
        files: List[str]) -> Dict[str, str]:

    for file in tqdm(files):
        filepath = opj(repository, file)

        # TODO - more functions for different file types
        if not filepath.endswith('.py'):
            continue

        # TODO - remove
        if 'llmtools' not in filepath:
            continue

        file_functions = extract_python_functions_from_file(filepath)

        for function_name, function_info in file_functions.items():
            function_text = function_info['text']
            function_id = file + '::' + function_name

            if function_id in function_descriptions:
                continue

            prompt = 'Generate a human readable description for the following Python function.\n'
            prompt += 'Function ID: ' + function_id
            prompt += 'Function Text:\n' + function_text
            function_descriptions[function_id] = f'Function_id: {function_id}\n'
            function_descriptions[function_id] += llm(prompt)

    return function_descriptions


def find_object_use(
        filepaths: List[str],
        package_module_name: str,
        object_name: str
    ) -> Dict[str, List[str]]:
    """Returns a dictionary of filepaths to lines where the object is called."""
    function_use = {}

    for file in tqdm(filepaths):
        with open(file, 'r') as f:
            file_text = f.read()
        imports = get_imports_from_code(file_text)
        function_call = import_called_as(imports, package_module_name, object_name)

        if function_call is None:
            continue

        function_calls = [
            line.strip()
            for line in file_text.split('\n')
            if function_call in line
        ]
        if len(function_calls) > 0:
            function_use[file] = function_calls

    return function_use



def init_db(client: chromadb.Client, repository: str) -> tuple[chromadb.Collection, list[str]]:

    # Get the committed files from the repo
    files = list(get_committed_files(repository, exts))
    files = [x.split(os.path.abspath(repository))[1][1:] for x in files]

    collection = client.get_or_create_collection(
        name='database', embedding_function=DEFAULT_EF
    )

    return collection, files

