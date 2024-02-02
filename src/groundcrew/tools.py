"""
File for Tools
"""
import os
import subprocess
import re

from typing import Callable

from thefuzz import process as fuzzprocess
from chromadb import Collection

from groundcrew import code, system_prompts as sp, code_utils as cu
from groundcrew.dataclasses import Chunk


def query_codebase(
        prompt: str, collection: Collection, n_results: int = 5, where: dict = None):
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


def get_paths(collection: Collection) -> dict[str, str]:
    """Get a dict filepaths (keyed by id) from the collection's metadata."""

    # get all paths and ids
    all_entries = collection.get(
        include=['metadatas']
    )

    return {
        x: y['filepath']
        for x, y in zip(all_entries['ids'], all_entries['metadatas'])
    }


def fuzzy_match_file_path(
        collection: Collection,
        search: str,
        thresh: int
    ) -> str | None:
    """Find a real file path in a collection given an example."""

    # there's a limited number of metadata filter options in chroma
    # so we'll grab everything and manufally filter

    paths = list(set(get_paths(collection).values()))

    # it might be possible to do something where we fuzzy match on ids instead
    # and then we could filter result lines by chunk

    # fuzzy match on filepaths
    top, thresh_match = fuzzprocess.extractOne(search, paths)

    if thresh_match < thresh:
        return None
    return top


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
        filepath = fuzzy_match_file_path(self.collection, filepath_inexact, 50)

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
        """

        if get_filename_from_id(id_) == filename or id_ == filename:
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

    def __call__(
            self,
            user_prompt: str,
            filepath_inexact: str = 'none',
            sort_on: str = 'max'
        ) -> str:
        """Answer questions using about the complex parts of a codebase.
        This method will create a summary of the most complex files to help answer
        the question.

        Args:
            user_prompt (str): The user's question.
            filepath_inexact (str): A filepath explicitly requested by the user.  If the
              user did not explicitly request a filepath, pass 'none'.
            sort_on (str): Sort the results by the "average" or "max" complexity of the file.
        """

        if filepath_inexact != 'none':
            filepath = fuzzy_match_file_path(self.collection, filepath_inexact, 50)

            if filepath is None:
                return f'Could not find a source file matching `{filepath_inexact}`'

            all_items = self.collection.get(
                include=['metadatas'],
                where={'filepath': filepath}
            )

            files = {
                id_: metadata['text']
                for id_, metadata in zip(all_items['ids'], all_items['metadatas'])
                if id_[-3:] == '.py'
            }

        else:
            files = get_python_files(self.collection)

        sort_on = str(sort_on).lower()
        if sort_on not in ['average', 'max']:
            return 'The sort_on parameter passed to CyclomaticComplexityTool must be "average" or "max".'

        complexity_summary_str = self.complexity_analysis(files, sort_on)

        prompt = (
            complexity_summary_str +
            '\n### Task ###\n' + self.base_prompt +
            '\n### Question ###\n' + user_prompt + '\n'
        )

        return self.llm(prompt)

    @staticmethod
    def __get_complexity(source_code: str) -> dict[str, dict]:
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

    def complexity_analysis(self, files: dict[str, str], sort_on: str) -> tuple[list[str], dict]:
        """Analyze the complexity of the codebase.

        This method will return a list of the most complex files in the codebase.
        If there are objects exceeding the `min_max_complexity`, then only those
        objects will be returned.  Otherwise, the list will return a "greedy" search
        for the most complex objects, including only those more complex than any found
        prior to it."""

        file_complexity = {}
        current_max = 0
        for file, source_code in files.items():
            complexity = self.__get_complexity(source_code)
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

        return self.__summarize(file_complexity, sorted_filenames)

    @staticmethod
    def __summarize(file_complexity: dict, file_list: list[str]) -> str:
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

        import_pattern = r'^import\s|from.*\simport\s'

        usages = {}
        for file, source_code in files.items():
            file_imports = cu.get_imports_from_code(source_code)
            if cu.imports_entity(file_imports, importable_object):
                called_as = cu.import_called_as(file_imports, importable_object)
                source_without_imports = '\n'.join(
                    line for line in source_code.split('\n')
                    if not re.search(import_pattern, line)
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


class GetFileContentsTool:
    """
    Interact with the contents of a specific file using natural language.
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
        self.base_prompt = base_prompt
        self.working_dir_path = working_dir_path

    def __call__(
            self,
            user_prompt: str,
            filepath_inexact: str) -> str:
        """
        Answer questions about a specific file.
        filepath_inexact is a file path which can be inexact, it will be fuzzy
        matched to find an exact file path for the project.
        """

        # ensure that filepath is a real path of a file in the collection
        filepath = fuzzy_match_file_path(self.collection, filepath_inexact, 50)

        if filepath is None:
            return f'Could not find a source file matching `{filepath_inexact}`'

        items = self.collection.get(
            include=['metadatas'],
            where={'filepath': filepath, 'type': 'file'}
        )

        output = (
            f'Here is file {items["metadatas"][0]["filepath"]}:\n\n' +
            items['metadatas'][0]['text']
        )

        return output


class InstallationAndUseTool:
    """
    This tool answers questions about the installation and execution of
    the codebase by querying for documentation files.
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
        self.base_prompt = base_prompt
        self.working_dir_path = working_dir_path

    def __call__(self, user_prompt: str) -> str:
        """Run the tool.

        Args:
            user_prompt (str): The prompt to process.
        """
        query = (
            "documentation regarding the installation and use of the codebase. "
            "README, configuration, environment."
        )
        doc_files = query_codebase(query, self.collection, n_results=15, where={'type': 'file'})
        doc_files_uids = [chunk.uid for chunk in doc_files if '..' not in chunk.filepath]

        results = query_codebase(
            user_prompt,
            self.collection, n_results=10,
            where={'id': {'$in': doc_files_uids}}
        )

        prompt = self.base_prompt + '\n\n'
        for chunk in results:
            prompt += f'### Contents of file {chunk.filepath} ###\n'
            prompt += chunk.text + '\n\n'

        prompt += self.base_prompt + '\n### Question ###\n'
        prompt += f'{user_prompt}\n\n'

        return self.llm(prompt)
