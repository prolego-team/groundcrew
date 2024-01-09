"""
File for Tools
"""
from abc import ABC, abstractmethod
from typing import Callable

from chromadb.api.models.Collection import Collection

from groundcrew import code
from groundcrew.dataclasses import Chunk


class ToolBase(ABC):

    def __init__(self, base_prompt: str, collection, llm):
        """ """
        self.base_prompt = base_prompt
        self.collection = collection
        self.llm = llm

    @abstractmethod
    def __call__(self, prompt: str, **kwargs):
        pass


class CodebaseQATool(ToolBase):

    def __init__(self, base_prompt: str, collection, llm):
        """ """
        super().__init__(base_prompt, collection, llm)

    def __call__(self, prompt: str, include_code: bool):

        chunks = self.query_codebase(prompt)

        prompt = self.base_prompt + '### Question ###\n'
        prompt += f'{prompt}\n\n'

        for chunk in chunks:
            prompt += code.format_chunk(chunk, include_text=include_code)
            prompt += '--------\n\n'

        return self.llm(prompt)

    def query_codebase(self, prompt: str, n_results: int=5):

        out = self.collection.query(
            query_texts=[prompt],
            n_results=n_results,
            where={'type': 'function'}
        )

        return [
            Chunk(
                name=metadata['name'],
                uid=metadata['id'],
                typ='function',
                text=metadata['function_text'],
                document=doc,
                filepath=metadata['filepath'],
                start_line=metadata['start_line'],
                end_line=metadata['end_line']
            ) for id_, metadata, doc in zip(
                out['ids'][0], out['metadatas'][0], out['documents'][0]
                )
        ]

