"""
Main run script
"""
import os
import ast
import pickle

from typing import Callable

import yaml
import click
import chromadb

from chromadb.api.models.Collection import Collection

from groundcrew import system_prompts as sp, utils
from groundcrew.code import extract_python_from_file, init_db
from groundcrew.agent import Agent
from groundcrew.dataclasses import Config

opj = os.path.join
CLASS_NODE_TYPE = ast.ClassDef
FUNCTION_NODE_TYPE = ast.FunctionDef


def populate_db(descriptions: dict[str, str], collection: Collection):
    """
    Populate a database with metadata and descriptions of Python functions
    extracted from a list of files.

    Args:
        descriptions (dict): A dictionary mapping file, function, or class
        identifiers to their LLM generated descriptions.
        collection (object): The database collection where data is to be
        upserted.
    """
    ids = []
    metadatas = []
    documents = []

    # Create ids, metadata, and documents
    # Name is a unique identifier
    for name, info in descriptions.items():

        # Choose the correct data type
        data_type = 'file'
        if name.endswith(' (class)'):
            data_type = 'class'
        elif name.endswith(' (function)'):
            data_type = 'function'

        filepath = name.split('::')[0]

        # Create metadata
        metadata = {
            'type': data_type,
            'id': name,
            'filepath': filepath,
            'text': info['text'],
            'start_line': info['start_line'],
            'end_line': info['end_line'],
            'summary': info['summary']
        }

        ids.append(name)
        metadatas.append(metadata)

        # The document is the LLM generated summary
        documents.append(info['summary'])

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )


def summarize_file(
        filepath: str,
        llm: Callable,
        descriptions: dict):
    """
    Function to summarize a file. If the file is a python file, it will search
    for functions and classes and summarize those as well.
    """

    # TODO - skip file if there are too many tokens

    code_dict = {}

    # Get the file text
    with open(filepath, 'r') as f:
        file_text = ''.join(f.readlines())

    # If it's a Python file also extract classes and functions
    if filepath.endswith('.py'):

        # Summarize entire Python file
        if filepath not in descriptions:
            print('Generating summary for', filepath, '...')
            prompt = filepath + '\n\n' + file_text + '\n\n'
            prompt += sp.SUMMARIZE_CODE_PROMPT
            file_summary = llm(prompt)
            descriptions[filepath] = {
                'text': file_text,
                'start_line': 1,
                'end_line': len(file_text.split('\n')),
                'summary': file_summary
            }
        else:
            print('Loading summary for', filepath, '...')

        # Extract classes and functions
        classes_dict = extract_python_from_file(file_text, CLASS_NODE_TYPE)
        functions_dict = extract_python_from_file(file_text, FUNCTION_NODE_TYPE)

        classes_dict = {k + ' (class)': v for k, v in classes_dict.items()}
        functions_dict = {
            k + ' (function)': v for k, v in functions_dict.items()
        }

        # Combine classes and functions dictionaries
        code_dict = {**classes_dict, **functions_dict}

        # Generate summaries for classes and functions
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

    # Not a Python file
    else:
        key = filepath
        if key not in descriptions:
            print('Generating summary for', key, '...')
            prompt = filepath + '\n\n' + file_text + '\n\n'
            prompt += sp.SUMMARIZE_FILE_PROMPT
            file_summary = llm(prompt)
            descriptions[key] = {
                'text': file_text,
                'start_line': 1,
                'end_line': len(file_text.split('\n')),
                'summary': file_summary
            }
        else:
            print('Loading summary for', key)


@click.command()
@click.option('--config', '-c', default='config.yaml')
@click.option('--model', '-m', default='gpt-4-1106-preview')
def main(config: str, model: str):
    """
    Main run script

    Args:
        config (str): Path to the config yaml file
        model (str): The name of the LLM model to use
    """

    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    config = Config(**config)

    # Directory to store generated file and function descriptions
    os.makedirs(config.cache_dir, exist_ok=True)

    # Create the chromadb client
    client = chromadb.PersistentClient(config.db_path)

    # Initialize the database and get a list of files in the repo
    collection, files = init_db(client, config.repository, config.extensions)
    files = sorted(files)

    # LLM that takes a string as input and returns a string
    llm = utils.build_llm_client(model)

    # File for storing LLM generated descriptions of files, functions, and
    # classes
    descriptions_file = opj(config.cache_dir, 'descriptions.pkl')

    # Dictionary storing LLM generated summaries.
    # If the file is a Python file, key will be filename :: class/function name
    # If the file is not a Python file, key will be the filename
    descriptions = {}
    if os.path.exists(descriptions_file):
        with open(descriptions_file, 'rb') as f:
            descriptions = pickle.load(f)

    # Generate summaries for files, classes, and functions
    for i, filepath in enumerate(files):
        filepath = opj(config.repository, filepath)
        summarize_file(filepath, llm, descriptions)

        # TODO - remove before merging
        if i > 5:
            break

    # Save the descriptions to a file in the cache directory
    with open(descriptions_file, 'wb') as f:
        pickle.dump(descriptions, f)

    # Populate the database with the files and descriptions
    populate_db(
        descriptions,
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

