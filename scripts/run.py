"""
Main run script
"""
import os
import pickle

from typing import Dict, List

import yaml
import click
import chromadb

from groundcrew import utils
from groundcrew.code import (extract_python_functions_from_file,
                             generate_function_descriptions, init_db)
from groundcrew.agent import Agent
from groundcrew.dataclasses import Config

opj = os.path.join


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

    # LLM that takes a string as input and returns a string
    llm = utils.build_llm_client(model)

    # Load or generate Tools
    tools_filepath = opj(config.cache_dir, 'tools.yaml')
    tool_descriptions = utils.setup_and_load_yaml(tools_filepath, 'tools')
    tools = utils.setup_tools(
        config.Tools,
        tool_descriptions,
        collection,
        llm)
    utils.save_tools_to_yaml(tools, tools_filepath)

    # File for storing LLM generated descriptions
    function_descriptions_file = opj(
        config.cache_dir, 'function_descriptions.pkl')

    if os.path.exists(function_descriptions_file):
        with open(function_descriptions_file, 'rb') as f:
            function_descriptions = pickle.load(f)
    else:
        # TODO - also run this if we want to force update descriptions
        function_descriptions = generate_function_descriptions(
            llm=llm,
            function_descriptions={},
            repository=config.repository,
            files=files
        )

        with open(function_descriptions_file, 'wb') as f:
            pickle.dump(function_descriptions, f)

    # Populate the database with the files and descriptions
    populate_db(
        config.repository,
        files,
        function_descriptions,
        collection
    )

    agent = Agent(config, collection, llm, tools)

    agent.run()


if __name__ == '__main__':
    main()

