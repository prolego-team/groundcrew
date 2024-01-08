"""
Evaluation framework.
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Ben Zimmer

from typing import Any, Dict, List, Optional, Tuple
import os
import re

import chromadb
import click
import tqdm
import yaml

from groundcrew.dataclasses import Config
from groundcrew import tools
from groundcrew import code

from scripts import run


@click.command()
@click.option('--config', '-c', default='config.yaml')
@click.option('--model', '-m', default='gpt-3.5-turbo')
@click.option('--evals', '-e', default='data/evaluation.yaml')
def main(config, model, evals):
    """main program"""

    # For now we'll assume that `run.py` has been run
    # and the database is present

    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    config = Config(**config)

    with open(evals, 'r') as f:
        evals = yaml.safe_load(f)

    # Directory to store generated file and function descriptions
    os.makedirs(config.cache_dir, exist_ok=True)

    # Create the chromadb client
    client = chromadb.PersistentClient(config.db_path)

    collection = client.get_or_create_collection(
        name=code.DEFAULT_COLLECTION_NAME,
        embedding_function=code.DEFAULT_EF
    )

    llm = run.build_llm_client(model)

    tools_dict = dict(
        codebase_qa=tools.codebase_qa
    )

    eval_funcs_dict = dict(
        match_word_any=match_word_any
    )

    # Run evaluations

    for suite in evals:
        run_suite(
            suite,
            eval_funcs_dict,
            collection,
            llm,
            tools_dict,
        )

        # TODO: proper returns


def run_suite(suite: Dict, eval_funcs_dict, collection, llm, tools_dict) -> None:
    """run a test suite and print the results"""
    print(suite)
    print(suite['name'])
    res = []
    for test in tqdm.tqdm(suite['tests']):
        tool = tools_dict[test['tool']]
        question = test['question']
        eval_params = dict(test['eval_func'])
        eval_func = eval_funcs_dict[eval_params['typ']]
        del eval_params['typ']

        answer = tool(question, collection, llm)
        res_test = eval_func(answer, **eval_params)
        res.append(res_test)

    print(sum(res), '/', len(res), 'passed')
    print('~~~~ ~~~~ ~~~~ ~~~~')


def match_word_any(text: str, words: List[str], strip_quotes: bool) -> bool:
    """
    Check if words in text match one or more of a list of words.
    Optionally strip quotes, which is useful for checking keywords
    like function names where the LLM will probably surround them
    with some type of quote.
    """
    quote_types_to_strip = '\'"`'

    text_words = re.split('\\s+', text)
    if strip_quotes:
        words = [x.strip(quote_types_to_strip) for x in text_words]
    for word in words:
        if word in text_words:
            return True
    return False


if __name__ == '__main__':
    main()
