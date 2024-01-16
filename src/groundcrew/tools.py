"""
File for Tools
"""
from typing import Callable

from chromadb import Collection

from groundcrew import code
from groundcrew.dataclasses import Chunk


def query_codebase(
        prompt: str, collection: Collection, n_results: int=5, where: dict=None):
    """
    Queries the codebase for relevant code chunks based on a given prompt.

    Args:
        prompt (str): The prompt to query the codebase.
        n_results (int, optional): The number of results to return.
        Defaults to 5.

    Returns:
        list: A list of code chunks relevant to the prompt.
    """
    out = collection.query(
        query_texts=[prompt],
        n_results=n_results,
        where=where
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


class CodebaseQATool:
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
        self.base_prompt = base_prompt
        self.collection = collection
        self.llm = llm

    def __call__(self, prompt: str, include_code: bool) -> str:
        """
        Processes a given prompt, queries the codebase, and uses the language
        model to generate a response.

        Args:
            prompt (str): The prompt to process.
            include_code (bool): Flag to include code in the response.

        Returns:
            str: The generated response from the language model.
        """
        chunks = query_codebase(prompt, self.collection)

        prompt = self.base_prompt + '### Question ###\n'
        prompt += f'{prompt}\n\n'

        for chunk in chunks:
            prompt += code.format_chunk(chunk, include_text=include_code)
            prompt += '--------\n\n'

        return self.llm(prompt)


