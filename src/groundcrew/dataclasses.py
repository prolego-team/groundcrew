"""
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    repository: str
    db_path: str
    cache_dir: str


@dataclass
class Chunk:
    """
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


