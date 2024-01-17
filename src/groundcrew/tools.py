"""
File for Tools
"""
import os

from abc import ABC, abstractmethod
from typing import Callable

from chromadb import Collection

from groundcrew import code, system_prompts as sp
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


def get_filename_from_id(id_: str):
    """
    Gets the filename from the ID used in the database.

    Args:
        id_ (str): The ID to parse.

    Returns:
        str: The filename.
    """
    return os.path.basename(id_.split('::')[0])


class SingleDocstringTool:
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
        self.base_prompt = base_prompt
        self.collection = collection
        self.llm = llm

        # Adding additional instructions
        self.base_prompt = base_prompt + sp.DOCSTRING_PROMPT

    def __call__(
            self,
            user_prompt: str,
            filename: str = 'none',
            function_name: str = 'none') -> str:
        """
        Generate docstrings for a given function, or all functions in a given
        file.

        Scenarios:
            - filename is not 'none', function_name is 'none': generate
              docstrings for all functions in the file
            - filename is not 'none', function_name is not 'none': search for
              the correct function in the correct file and generate docstrings
            - filename is 'none', function_name is not 'none': search for the
              correct function in the database and generate docstring
            - filename is 'none', function_name is 'none': assumes the
              user_prompt includes code and generates the docstring for that
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
        if function_name == 'none':
            all_functions = True

        # IDs of the files/functions to generate docstrings for
        filtered_ids = []
        if filename != 'none':
            for id_ in all_ids:

                # This will match with all functions in the given file
                if all_functions and '::' in id_ and get_filename_from_id(id_) == filename:
                    filtered_ids.append(id_)

                # This will match with a single function in the given file
                elif not all_functions and f'::{function_name}' in id_ and get_filename_from_id(id_) == filename:
                    filtered_ids.append(id_)

        # No filename was given, find the function(s) given
        elif filename == 'none' and function_name != 'none':
            for id_ in all_ids:

                # If '::' isn't in the ID then it's a file
                if '::' not in id_:
                    continue

                if function_name in id_:
                    filtered_ids.append(id_)

        # User included code in their prompt so no filename or function needed
        if filename == 'none' and function_name == 'none':
            function_code = [user_prompt]
        else:
            function_code = []
            for id_ in filtered_ids:
                item = self.collection.get(id_)
                function_code.append(item['metadatas'][0]['text'] + '\n')

        function_code = '\n'.join(function_code)
        prompt = function_code + '\n### Task ###\n' + self.base_prompt + '\n'

        return self.llm(prompt)


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

    def __call__(self, user_prompt: str, include_code: bool) -> str:
        """
        Processes a given prompt, queries the codebase, and uses the language
        model to generate a response.

        Args:
            prompt (str): The prompt to process.
            include_code (bool): Flag to include code in the response.

        Returns:
            str: The generated response from the language model.
        """
        chunks = query_codebase(user_prompt, self.collection)

        prompt = ''
        for chunk in chunks:
            #print(chunk.text)
            #exit()
            prompt += code.format_chunk(chunk, include_text=include_code)
            prompt += '--------\n\n'

        prompt += self.base_prompt + '\n### Question ###\n'
        prompt += f'{user_prompt}\n\n'

        print(prompt)

        return self.llm(prompt)

