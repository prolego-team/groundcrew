"""
File for Tools
"""
import os

from abc import ABC, abstractmethod
import subprocess
from typing import Callable

from chromadb.api.models.Collection import Collection

from groundcrew import code, system_prompts as sp
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

    def get_filename_from_id(self, id_: str):
        return os.path.basename(id_.split('::')[0])

    def query_codebase(self, prompt: str, n_results: int=5, where: dict=None):
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

    @abstractmethod
    def __call__(self, prompt: str, **kwargs):
        """
        The call method
        """
        pass


class LinterTool(ToolBase):
    """
    Interact with a linter using natural language.
    """

    def __init__(self, base_prompt: str, collection: Collection, llm: Callable):
        """Constructor."""
        super().__init__(base_prompt, collection, llm)
        self.base_prompt = base_prompt + sp.LINTER_PROMPT
        self.working_dir_path: str | None = None

    def __call__(
            self,
            prompt: str,
            filepath: str) -> str:
        """
        Answer questions using about linting results for a file.
        Linters usually operate per file so this granularity makes sense.
        """

        # TODO: do a fuzzy match of some kind on the filepath to find an actual filepath

        try:
            command = ['ruff', '--preview',  filepath]
            print(command)
            print(self.working_dir_path)
            linter_output = subprocess.check_output(command, cwd=self.working_dir_path)
        except subprocess.CalledProcessError as e:
            linter_output = e.output

        linter_output = str(linter_output)

        prompt = linter_output + '\n### Task ###\n' + self.base_prompt + '\n'

        return self.llm(prompt)


class SingleDocstringTool(ToolBase):
    """
    """
    def __init__(self, base_prompt: str, collection: Collection, llm: Callable):
        """
        Initialize the SingleDocstringTool with a base prompt, a code
        collection, and a language model.

        Args:
            base_prompt (str): The base prompt to prepend to all queries.
            collection (Collection): The code collection or database to query
            for code-related information.
            llm (Callable): The language model to use for generating
            code-related responses.
        """
        super().__init__(base_prompt, collection, llm)

        # Adding additional instructions
        self.base_prompt = base_prompt + sp.DOCSTRING_PROMPT

    def __call__(
            self,
            prompt: str,
            filename: str = None,
            function_name: str = None):
        """
        Generate docstrings for a given function, or all functions in a given
        file.

        Scenarios:
            - filename is not None, function_name is None: generate docstrings
              for all functions in the file
            - filename is not None, function_name is not None: generate
              docstring for single function in the file
            - filename is None, function_name is not None: generate docstring
              for single function

        Args:
            prompt (str): The prompt to process.
            filename (str): A filename to query and generate docstrings for all
            functions within the file. If empty, pass "none".
            function_name (str): The name of the function to generate a
            docstring for. If empty, pass "none".

        Returns:
            str: The generated response from the language model.
        """

        all_ids = self.collection.get()['ids']

        # Flag for generating docstrings for all functions in a file
        all_functions = False
        if function_name == 'none' or function_name is None:
            all_functions = True

        # IDs of the files/functions to generate docstrings for
        filtered_ids = []
        if filename != 'none':
            for id_ in all_ids:

                # This will match with all functions in the given file
                if all_functions and '::' in id_ and self.get_filename_from_id(id_) == filename:
                    filtered_ids.append(id_)

                # This will match with a single function in the given file
                elif not all_functions and f'::{function_name}' in id_ and self.get_filename_from_id(id_) == filename:
                    filtered_ids.append(id_)

        # No filename was given, find the function(s) given
        elif filename == 'none':
            for id_ in all_ids:

                # If '::' isn't in the ID then it's a file
                if '::' not in id_:
                    continue

                if function_name in id_:
                    filtered_ids.append(id_)

        function_code = []
        for id_ in filtered_ids:
            item = self.collection.get(id_)
            function_code.append(item['metadatas'][0]['text'] + '\n')

        function_code = '\n'.join(function_code)
        prompt = function_code + '\n### Task ###\n' + self.base_prompt + '\n'

        return self.llm(prompt)


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

