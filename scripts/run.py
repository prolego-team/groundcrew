"""
Main run script
"""
import os
import ast
import pickle

from typing import Dict, List

import yaml
import click
import chromadb

from groundcrew import system_prompts as sp, utils
from groundcrew.code import extract_python_from_file, init_db
from groundcrew.agent import Agent
from groundcrew.dataclasses import Config

opj = os.path.join
CLASS_NODE_TYPE = ast.ClassDef
FUNCTION_NODE_TYPE = ast.FunctionDef


def populate_db(
        repository: str,
        files: List[str],
        function_descriptions: Dict,
        collection):
    """
    Populate a database with metadata and descriptions of Python functions
    extracted from a list of files.

    Args:
        repository (str): The base directory where the files are located.
        files (List[str]): A list of file names to be processed.
        function_descriptions (Dict): A dictionary mapping function identifiers
        to their descriptions.
        collection (object): The database collection where function data is to
        be upserted.
    """
    ids = []
    metadatas = []
    documents = []
    for file in files:
        filepath = opj(repository, file)

        # TODO - refactor to make more general
        if not filepath.endswith('.py'):
            continue

        file_functions = extract_python_functions_from_file(filepath)
        for function_name, function_info in file_functions.items():

            function_text = function_info['text']
            start_line = function_info['start_line']
            end_line = function_info['end_line']

            function_id = file + '::' + function_name

            function_description = function_descriptions.get(function_id)
            if function_description is None:
                continue

            metadata = {
                'type': 'function',
                'name': function_name,
                'id': function_id,
                'filepath': file,
                'function_text': function_text,
                'start_line': start_line,
                'end_line': end_line
            }

            ids.append(function_id)
            metadatas.append(metadata)
            documents.append(function_description)

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )



def summarize_file(
        filepath,
        llm,
        descriptions):
    """
    Function to summarize a file. If the file is a python file, it will search
    for functions and classes and summarize those as well.
    """

    code_dict = {}

    with open(filepath, 'r') as f:
        file_text = ''.join(f.readlines())

    if filepath.endswith('.py'):

        prompt = filepath + '\n\n' + file_text + '\n\n'
        prompt += sp.SUMMARIZE_CODE_PROMPT
        file_summary = llm(prompt)

        classes_dict = extract_python_from_file(file_text, CLASS_NODE_TYPE)
        functions_dict = extract_python_from_file(file_text, FUNCTION_NODE_TYPE)

        classes_dict = {k + ' (class)': v for k, v in classes_dict.items()}
        functions_dict = {
            k + ' (function)': v for k, v in functions_dict.items()
        }

        code_dict = {**classes_dict, **functions_dict}
        new_code_dict = {}
        for name, info in code_dict.items():

            key = filepath + '::' + name

            if key not in descriptions:
                print('Generating summary for', key, '...')
                prompt = filepath + '\n\n' + info['text'] + '\n\n'
                prompt += sp.SUMMARIZE_CODE_PROMPT
                info['summary'] = llm(prompt)
                descriptions[key] = info
            else:
                print('Loading summary for', key)
            new_code_dict[key] = info

        code_dict = new_code_dict

    else:
        key = filepath
        if key not in descriptions:
            prompt = filepath + '\n\n' + file_text + '\n\n'
            prompt += sp.SUMMARIZE_FILE_PROMPT
            file_summary = llm(prompt)
            descriptions[key] = file_summary
        else:
            file_summary = descriptions[key]

    return file_summary, code_dict


@click.command()
@click.option('--config', '-c', default='config.yaml')
@click.option('--model', '-m', default='gpt-3.5-turbo')
def main(config, model):
    """
    Main
    """

    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    config = Config(**config)

    # Directory to store generated file and function descriptions
    os.makedirs(config.cache_dir, exist_ok=True)

    # Create the chromadb client
    client = chromadb.PersistentClient(config.db_path)

    # Initialize the database and get a list of files in the repo
    collection, files = init_db(client, config.repository)
    files = sorted(files)

    # LLM that takes a string as input and returns a string
    llm = utils.build_llm_client(model)

    # File for storing LLM generated descriptions
    descriptions_file = opj(
        config.cache_dir, 'descriptions.pkl')

    # Dictionary of filename :: class/function name
    descriptions = {}
    if os.path.exists(descriptions_file):
        with open(descriptions_file, 'rb') as f:
            descriptions = pickle.load(f)

    for filepath in files:
        filepath = opj(config.repository, filepath)
        if not filepath.endswith('.py'):
            continue
        file_summary, code_dict = summarize_file(filepath, llm, descriptions)
        exit()

    with open(function_descriptions_file, 'wb') as f:
        pickle.dump(function_descriptions, f)
    # Populate the database with the files and descriptions
    populate_db(
        config.repository,
        files,
        function_descriptions,
        collection
    )

    # Load or generate Tools
    tools_filepath = opj(config.cache_dir, 'tools.yaml')
    tool_descriptions = utils.setup_and_load_yaml(tools_filepath, 'tools')
    tools = utils.setup_tools(
        config.Tools,
        tool_descriptions,
        collection,
        llm)
    utils.save_tools_to_yaml(tools, tools_filepath)
    agent = Agent(config, collection, llm, tools)

    agent.run()


if __name__ == '__main__':
    main()

