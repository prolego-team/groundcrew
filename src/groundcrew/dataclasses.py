"""
"""
from typing import Callable

import yaml

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Class to hold config parameters
    """
    repository: str
    extensions: list[str]
    db_path: str
    cache_dir: str
    Tools: [str]
    colorscheme: str
    debug: bool
    shell: str
    conda_dir: str

@dataclass
class Chunk:
    """
    Class to hold text data such as a python function, text file, etc.

    if typ is 'function' or 'class':
        document is the function/class summary
        text is the function code

    if typ is 'file':
        document is the file summary
        text is the entire file text
    """
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
    Class to hold a Tool which is a Python class that the agent can use
    """
    name: str
    code: str
    description: str
    base_prompt: str
    params: dict
    obj: Callable

    def to_yaml(self):
        data = {
            'name': self.name,
            'description': self.description,
            'base_prompt': self.base_prompt,
            'params': self.params
        }
        return yaml.dump(
            data, default_flow_style=False, sort_keys=False)

    def to_string(self):
        output = f'\nTool Name: {self.name}\n'
        output += f'Description: {self.description}\n'
        output += f'Base Prompt: {self.base_prompt}\n'
        return output


class Colors:
    """ Colors for printing """
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[94m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    ENDC = '\033[0m'
