"""
File for Tools
"""
from abc import ABC, abstractmethod
from typing import Callable

from chromadb.api.models.Collection import Collection
from radon.visitors import ComplexityVisitor

from groundcrew import code
from groundcrew.dataclasses import Chunk


class ToolBase(ABC):
    """
    Abstract base class for tools that interact with a language model (LLM).

    Attributes:
        base_prompt (str): The base prompt used for interactions with the LLM.
        collection (Collection): A chromaDB collection
        llm (Callable): The language model instance used for generating
        responses.

    Methods:
        __call__(prompt: str, **kwargs): Abstract method to be implemented in
        subclasses.
    """
    def __init__(self, base_prompt: str, collection: Collection, llm: Callable):
        """
        Constructor
        """
        self.base_prompt = base_prompt
        self.collection = collection
        self.llm = llm

    @abstractmethod
    def __call__(self, prompt: str, **kwargs):
        """
        The call method
        """
        pass


class CodebaseQATool(ToolBase):
    """
    Tool for querying a codebase and generating responses using a language
    model.  Inherits from ToolBase and implements the abstract methods for
    specific codebase querying functionality.
    """
    def __init__(self, base_prompt: str, collection: Collection, llm: Callable):
        """
        Initialize the CodebaseQATool with a base prompt, a code collection,
        and a language model.

        Args:
            base_prompt (str): The base prompt to prepend to all queries.
            collection (Collection): The code collection or database to query
            for code-related information.
            llm (Callable): The language model to use for generating
            code-related responses.
        """
        super().__init__(base_prompt, collection, llm)

    def __call__(self, prompt: str, include_code: bool):
        """
        Processes a given prompt, queries the codebase, and uses the language
        model to generate a response.

        Args:
            prompt (str): The prompt to process.
            include_code (bool): Flag to include code in the response.

        Returns:
            str: The generated response from the language model.
        """
        chunks = self.query_codebase(prompt)

        prompt = self.base_prompt + '### Question ###\n'
        prompt += f'{prompt}\n\n'

        for chunk in chunks:
            prompt += code.format_chunk(chunk, include_text=include_code)
            prompt += '--------\n\n'

        return self.llm(prompt)

    def query_codebase(self, prompt: str, n_results: int=5):
        """
        Queries the codebase for relevant code chunks based on a given prompt.

        Args:
            prompt (str): The prompt to query the codebase.
            n_results (int, optional): The number of results to return.
            Defaults to 5.

        Returns:
            list: A list of code chunks relevant to the prompt.
        """
        out = self.collection.query(
            query_texts=[prompt],
            n_results=n_results,
            #where={'type': 'function'} # TODO - might want the LLM to choose which types to query
        )

        return [
            Chunk(
                uid=metadata['id'],
                typ=metadata['type'],
                text=metadata['text'],
                document=doc,
                filepath=metadata['filepath'],
                start_line=metadata['start_line'],
                end_line=metadata['end_line']
            ) for id_, metadata, doc in zip(
                out['ids'][0], out['metadatas'][0], out['documents'][0]
                )
        ]


def cyclomatic_complexity(code: str) -> dict:
    """Compute the cyclomatic complexity of a piece of code."""
    v = ComplexityVisitor.from_code(code)
    output = {}
    for func in v.functions:
        output[func.name] = {
            'object': 'function',
            'complexity': func.complexity
        }

    for clss in v.classes:
        output[clss.name] = {
            'object': 'class',
            'complexity': clss.complexity,
            'methods': {}
        }
        for meth in clss.methods:
            output[clss.name]['methods'][meth.name] = {
                'complexity': meth.complexity
            }

    return output

