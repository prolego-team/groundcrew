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


def extract_python_from_file(file_text, node_types):
    file_lines = file_text.split('\n')
    texts = {}

    class NodeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class = None

        def update_texts(self, node, key):
            texts[key] = {
                'text': '\n'.join(file_lines[node.lineno - 1:node.end_lineno]),
                'start_line': node.lineno,
                'end_line': node.end_lineno,
                'is_method': self.current_class is not None,
                'is_class': isinstance(node, ast.ClassDef)
            }

        def visit_ClassDef(self, node):
            if ast.ClassDef == node_types:
                self.update_texts(node, node.name)
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = None

        def visit_FunctionDef(self, node):
            if ast.FunctionDef == node_types:
                key = f"{self.current_class}.{node.name}" if self.current_class else node.name
                self.update_texts(node, key)
            self.generic_visit(node)

    visitor = NodeVisitor()
    visitor.visit(ast.parse(file_text))

    return texts


def init_db(client, repository, exts):

    # Get the committed files from the repo
    files = list(get_committed_files(repository, exts))
    files = [x.split(os.path.abspath(os.path.expanduser(repository)))[1][1:] for x in files]

    collection = client.get_or_create_collection(
        name='database', embedding_function=DEFAULT_EF
    )

    return collection, files
