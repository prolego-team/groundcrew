"""
File for Tools
"""
import os
import subprocess

from typing import Callable

from thefuzz import process as fuzzprocess
from chromadb import Collection

from groundcrew import code, system_prompts as sp
from groundcrew.dataclasses import Chunk


def query_codebase(
        prompt: str, collection: Collection, n_results: int=5, where: dict=None):
    """
    Queries the codebase for relevant code chunks based on a given prompt.

    Args:
        prompt (str): The prompt to query the codebase.
        collection (Collection): The chromadb collection to query
        n_results (int, optional): The number of results to return.
        where (dict, optional): A dictionary of additional metadata query
        parameters.

    Returns:
        list: A list of Chunk objects
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


class LintFileTool:
    """
    Interact with a linter using natural language.
    """

    def __init__(
            self,
            base_prompt: str,
            collection: Collection,
            llm: Callable,
            working_dir_path: str):
        """Constructor."""
        self.collection = collection
        self.llm = llm
        self.base_prompt = base_prompt + sp.LINTER_PROMPT
        self.working_dir_path = working_dir_path

    def __call__(
            self,
            user_prompt: str,
            filepath_inexact: str) -> str:
        """
        Answer questions using about linting results for a file.
        filepath_inexact is a file path which can be inexact, it will be fuzzy
        matched to find an exact file path for the project.
        Linters usually operate per file so this granularity makes sense.
        """

        # ensure that filepath is a real path of a file in the collection
        # TODO: figure out what the correct threshold is here...
        #       probably higher than 50
        filepath = self.fuzzy_match_file_path(filepath_inexact, 50)

        if filepath is None:
            return f'Could not find a source file matching `{filepath_inexact}`'

        linter_output = self.run_ruff(filepath)

        if not linter_output:
            linter_output = 'Linter did not find any issues.'

        prompt = (
            linter_output +
            '\n### Task ###\n' + self.base_prompt +
            '\n### Question ###\n' + user_prompt + '\n'
        )

        return self.llm(prompt)

    def run_ruff(self, filepath: str) -> str:
        """
        Run ruff on a file and capture the output.
        Filter some of the output like comments about things being fixable with `--fix`.
        """

        try:
            command = ['ruff', 'check', '--preview',  filepath]
            linter_output = subprocess.check_output(command, cwd=self.working_dir_path)
        except subprocess.CalledProcessError as e:
            linter_output = e.output

        linter_output = linter_output.decode()

        linter_output = [
            x for x in linter_output.split('\n')
            if not x.startswith('[*]')
        ]

        return '\n'.join(linter_output)

    def fuzzy_match_file_path(self, search: str, thresh: int) -> str | None:
        """Find a real file path in a collection given an example."""

        # there's a limited number of metadata filter options in chroma
        # so we'll grab everything and manufally filter

        paths = list(set(self.get_paths().values()))

        # it might be possible to do something where we fuzzy match on ids instead
        # and then we could filter result lines by chunk

        # fuzzy match on filepaths
        top, thresh_match = fuzzprocess.extractOne(search, paths)

        if thresh_match < thresh:
            return None
        return top

    def get_paths(self) -> dict[str, str]:
        """Get a dict filepaths (keyed by id) from the collection's metadata."""

        # get all paths and ids
        all_entries = self.collection.get(
            include=['metadatas']
        )

        return {
            x: y['filepath']
            for x, y in zip(all_entries['ids'], all_entries['metadatas'])
        }


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
            code: str,
            filename: str,
            function_name: str) -> str:
        """
        Generate docstrings for a given snippet of code, a function, or all
        functions in a given file.

        Scenarios:
            - code is not 'none': generate docstrings for the given code
            - filename is not 'none', function_name is 'none': generate
              docstrings for all functions in the file
            - filename is not 'none', function_name is not 'none': search for
              the correct function in the correct file and generate docstrings
            - filename is 'none', function_name is not 'none': search for the
              correct function in the database and generate docstring
            - filename is 'none', function_name is 'none': assumes the
              user_prompt includes code and generates the docstring for that
        Args:
            user_prompt (str): The prompt to process.
            code (str): The code to generate a docstring for.
            filename (str): A filename to query and generate docstrings for all
            functions within the file. If empty, pass "none".
            function_name (str): The name of the function to generate a
            docstring for. If empty, pass "none".

        Returns:
            str: The generated response from the language model.
        """

        # Get all of the IDs of the functions in the database
        all_ids = self.collection.get()['ids']

        # Determine the context based on input parameters
        context = self._determine_context(code, filename, function_name)

        if context == 'no_match':
            return 'No matching functions found.'

        # Get the function code we want to generate a docstring for
        function_code = self._get_function_code(
            all_ids, filename, function_name, code, context)

        prompt = function_code + '\n### Task ###\n' + self.base_prompt + '\n'
        return self.llm(prompt)

    def _determine_context(
            self, code: str, filename: str, function_name: str) -> str:
        if code != 'none':
            return 'code'
        if filename == 'none' and function_name != 'none':
            return 'function'
        if filename != 'none' and function_name != 'none':
            return 'file-function'
        if filename != 'none':
            return 'file'
        return 'no_match'

    def _get_function_code(
            self,
            all_ids: list[str],
            filename: str,
            function_name: str,
            code: str,
            context: str) -> str:

        function_code = []

        if context == 'code':
            return code

        filtered_ids = []
        for id_ in all_ids:

            # Adding all functions in the given file
            if context == 'file' and self._id_matches_file(
                    id_, filename, function_name):
                filtered_ids.append(id_)

            # Adding just the function that we're looking for
            elif context == 'function' and self._id_matches_function(
                    id_, function_name):
                filtered_ids.append(id_)

            # Adding a specific function in a specific file
            elif context == 'file-function' and self._id_matches_file(
                    id_, filename, function_name):
                filtered_ids.append(id_)

        for id_ in filtered_ids:
            item = self.collection.get(id_)
            function_code.append(item['metadatas'][0]['text'] + '\n')

        return '\n'.join(function_code)

    def _id_matches_file(
            self,
            id_: str,
            filename: str,
            function_name: str) -> bool:
        """
        Checks if an ID matches a given filename

        Args:
            id_ (str): The ID to check.
            filename (str): The filename to check against.
            function_name (str): The optional function name to check against.
            If function_name is 'none', then this will add all functions in the
            matched file
        """
        if get_filename_from_id(id_) == filename:
            return True if function_name == 'none' or f'::{function_name}' in id_ else False
        return False

    def _id_matches_function(self, id_: str, function_name: str) -> bool:
        return '::' in id_ and function_name in id_


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
        self.base_prompt = base_prompt + sp.CODEQA_PROMPT
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

        return self.llm(prompt)
