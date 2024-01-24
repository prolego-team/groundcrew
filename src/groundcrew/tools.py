"""
File for Tools
"""
import os

import subprocess
from typing import Callable

from chromadb.api.models.Collection import Collection
from thefuzz import process as fuzzprocess
from chromadb import Collection

from groundcrew import code, system_prompts as sp, code_utils as cu
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


def get_python_files(collection: Collection) -> dict[str, str]:
    """Get items from the collection"""
    all_items = collection.get(
        include=['metadatas'],
        where={'type': 'file'}
    )

    return {
        id_: metadata['text']
        for id_, metadata in zip(all_items['ids'], all_items['metadatas'])
        if id_[-3:] == '.py'
    }

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

        if not function_code:
            return 'No matching functions found.'

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


class CyclomaticComplexityTool:
    """
    Tool for computing the cyclomatic complexity of a codebase.
    """
    def __init__(
            self,
            base_prompt: str,
            collection: Collection,
            llm: Callable,
            min_max_complexity: int = 11
        ):
        self.collection = collection
        self.llm = llm
        self.base_prompt = base_prompt
        self.min_max_complexity = min_max_complexity

    def __call__(self, user_prompt: str, sort_on: str) -> str:
        """Answer questions using about the complex parts of a codebase.
        This method will create a summary of the most complex files to help answer
        the question."""

        sort_on = sort_on.lower()
        assert sort_on in ['average', 'max'], 'sort_on must be "average" or "max"'

        complexity_summary_str = self.complexity_analysis(sort_on)

        prompt = (
            complexity_summary_str +
            '\n### Task ###\n' + self.base_prompt +
            '\n### Question ###\n' + user_prompt + '\n'
        )

        return self.llm(prompt)

    def get_complexity(self, source_code: str) -> dict[str, dict]:
        """Get the complexity of source code."""

        complexity_dict = cu.cyclomatic_complexity(source_code)
        if len(complexity_dict) == 0:
            average_complexity, max_complexity = 0, 0
        else:
            average_complexity = sum(complexity_dict[obj]['complexity'] for obj in complexity_dict)/len(complexity_dict)
            max_complexity = max(complexity_dict[obj]['complexity'] for obj in complexity_dict)

        return {
            'average': average_complexity,
            'max': max_complexity,
            'details': complexity_dict
        }

    def complexity_analysis(self, sort_on: str) -> tuple[list[str], dict]:
        """Analyze the complexity of the codebase."""
        files = get_python_files(self.collection)

        file_complexity = {}
        current_max = 0
        for file, source_code in files.items():
            complexity = self.get_complexity(source_code)
            if complexity['max'] >= self.min_max_complexity or complexity['max'] > current_max:
                file_complexity[file] = complexity
                current_max = max(current_max, complexity['max'])

        sorted_complexity = sorted(file_complexity.items(), key=lambda x: x[1][sort_on], reverse=True)

        prune = (
            sorted_complexity[0][1]['max'] >= self.min_max_complexity and
            sorted_complexity[-1][1]['max'] < self.min_max_complexity
        )
        if prune:
            sorted_filenames = [
                item[0] for item in sorted_complexity
                if item[1]['max'] >= self.min_max_complexity
            ]
        else:
            sorted_filenames = [item[0] for item in sorted_complexity]

        return self.summarize(file_complexity, sorted_filenames)

    def summarize(self, file_complexity: dict, file_list: list[str]) -> str:
        """Summarize the complexity of a list of files.

        The file_complexity dict should be keyed by file name and have values that were
        generated by the get_complexity method.  file_list should be a _sorted_ list of
        file names that index decreasing complexity."""

        summary_str = 'Cyclomatic complexity summary for the most complex files:\n'
        for file in file_list:
            summary_str += (
                f'File: {file}; '
                f'average complexity = {file_complexity[file]["average"]}; '
                f'max complexity = {file_complexity[file]["max"]}\n'
            )
            for name, desc in file_complexity[file]['details'].items():
                if desc['object'] == 'function':
                    summary_str += f'  {name}: {desc["complexity"]}\n'
                elif desc['object'] == 'class':
                    summary_str += f'  {name}: {desc["complexity"]}\n'
                    for method, method_desc in desc['methods'].items():
                        summary_str += f'    {name}.{method}: {method_desc["complexity"]}\n'

        return summary_str


class FindUsageTool:

    def __init__(
            self,
            base_prompt: str,
            collection: Collection,
            llm: Callable
        ):
        self.collection = collection
        self.llm = llm
        self.base_prompt = base_prompt

    def __call__(self, user_prompt: str, importable_object: str) -> str:
        """Answer questions using about the usage of a given importable object.

        The importable_object should be the name of a module, function, class, or
        variable.  It should be a fully qualified name, e.g. 'numpy.random' or
        'numpy.random.randint'.

        This method will create a summary of the usage of the importable object to help
        answer the question.  Specifically, it will list the files that import the
        object and the number of times the object is used in each file.
        """
        usage = self.get_usage(importable_object)

        prompt = (
            f'Usage summary for {importable_object} (filename: usage count):\n' +
            self.summarize_usage(usage) +
            '\n### Task ###\n' + self.base_prompt +
            '\n### Question ###\n' + user_prompt + '\n'
        )

        return self.llm(prompt)

    def get_usage(self, importable_object: str) -> dict[str, int]:
        """Get the usage of an entity from a package/module."""
        files = get_python_files(self.collection)


        usages = {}
        for file, source_code in files.items():
            file_imports = cu.get_imports_from_code(source_code)
            if cu.imports_entity(file_imports, importable_object):
                called_as = cu.import_called_as(file_imports, importable_object)
                source_without_imports = '\n'.join(
                    line for line in source_code.split('\n')
                    if 'import' not in line
                )
                usage_count = sum(source_without_imports.count(call) for call in called_as)
                if usage_count > 0:
                    usages[file] = usage_count

        return usages

    def summarize_usage(self, usage: dict[int, str]) -> str:
        """Summarize a usage dict containing file names and usage counts."""
        summary_str = ''
        if len(usage) == 0:
            return summary_str + 'No usage found.\n'

        for file, count in usage.items():
            summary_str += f'{file}: {count}\n'

        return summary_str
