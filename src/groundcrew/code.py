"""
"""
import os
import ast

from typing import Callable, Dict, List

import chromadb
from git import Repo
from tqdm import tqdm

from groundcrew.constants import DEFAULT_EF

opj = os.path.join

DEFAULT_COLLECTION_NAME = 'database'


def get_committed_files(
        repository_dir: str, extensions: List[str]) -> List[str]:

    committed_files = set()
    repo = Repo(repository_dir)
    for blob in repo.head.commit.tree.traverse():
        file_path = os.path.join(repo.working_dir, blob.path)
        ext = os.path.splitext(file_path)[1]
        if ext in extensions:
            committed_files.add(file_path)

    return committed_files


def extract_python_functions_from_file(filepath):

    function_texts = {}

    with open(filepath, 'r') as f:
        file_lines = f.readlines()
        file_text = ''.join(file_lines)

    for node in ast.walk(ast.parse(file_text)):
        if isinstance(node, ast.FunctionDef):
            function_texts[node.name] = {
                'text': ''.join(
                    file_lines[node.lineno - 1:node.end_lineno]
                ),
                'start_line': node.lineno,
                'end_line': node.end_lineno
            }

    return function_texts


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


def init_db(client: chromadb.Client, repository: str) -> tuple[chromadb.Collection, list[str]]:

    # Get the committed files from the repo
    exts = ['.py', '.txt', '.yaml', '.yml', '.sh']
    files = list(get_committed_files(repository, exts))
    files = [x.split(os.path.abspath(repository))[1][1:] for x in files]

    collection = client.get_or_create_collection(
        name=DEFAULT_COLLECTION_NAME, embedding_function=DEFAULT_EF
    )

    return collection, files
