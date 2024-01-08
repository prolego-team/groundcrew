"""
"""
from typing import Callable

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    repository: str
    db_path: str
    cache_dir: str
    Tools: [str]


@dataclass
class Chunk:
    """
    Class to hold text data such as a python function, text file, etc.

    if typ is 'function':
        document is the function description
        text is the function code

    if typ is 'file':
        document is the file summary
        text is the entire file text
    """
    name: str
    uid: str
    typ: str
    text: str
    document: str
    filepath: str
    start_line: int
    end_line: int


@dataclass
class Tool:
    """
    Class to hold a Tool which is a Python function that the agent can use
    """
    name: str
    function_str: str
    description: str
    call: Callable

    def to_string(self):
        output = f'\nTool Name: {self.name}\n'
        output += f'Description: {self.description}\n'
        return output
