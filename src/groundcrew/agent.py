"""
Core agent functionality.
"""

from typing import Callable

import chromadb

from groundcrew.tools import codebase_qa
from groundcrew import dataclasses as dc


class Agent:
    """Core agent functionality."""

    def __init__(
            self,
            config: dc.Config,
            collection: chromadb.Collection,
            llm: Callable):

        self.config = config
        self.collection = collection
        self.llm = llm

    def run(self):

        while True:

            prompt = input('> ')
            if not prompt:
                prompt = 'What is the name of the function that finds pdfs in a directory?'

            response = codebase_qa(prompt, self.collection, self.llm)

            print(response)
            print('\n', 80 * '*', '\n')
