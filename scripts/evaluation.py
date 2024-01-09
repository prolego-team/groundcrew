"""
Evaluation framework.
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Ben Zimmer

from typing import Dict, List, Callable
import datetime
import os
import re
import time

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
@click.option('--n_runs', '-n', default=3)
@click.option('--output_dir_prefix', '-o', default='eval')
def main(config, model, evals, n_runs, output_dir_prefix):
    """main program"""

    # ~~~~ create output directory

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir_path = f'{output_dir_prefix}_{timestamp}'
    os.makedirs(output_dir_path, exist_ok=False)

    # ~~~~ build the system (collection, llm, etc)

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

    # OpenAI models can't be created with a seed
    # so this is a simple wrapper that ignores the seed
    llm_from_seed = lambda _: run.build_llm_client(model)

    tools_dict = dict(
        codebase_qa=tools.codebase_qa
    )

    eval_funcs_dict = dict(
        match_word_any=match_word_any
    )

    # ~~~~ run evaluations

    results = {}

    # TODO: runs across multiple systems

    for suite in evals:
        results_suite = run_suite(
            suite=suite,
            n_runs=n_runs,
            eval_funcs_dict=eval_funcs_dict,
            collection=collection,
            llm_from_seed=llm_from_seed,
            tools_dict=tools_dict,
        )
        results = {**results, **results_suite}

    output_file_prefix = 'eval'
    output_file_path = os.path.join(output_dir_path, f'{output_file_prefix}.csv')
    # header = 'suite,question,run,calls,time,missing,correct,answer\n'
    header = 'suite,question,run,time,missing,correct,answer\n'
    with open(output_file_path, 'w') as f:
        f.write(header)
        for (suite_name, question, run_idx), info in results.items():
            answer = info['answer']
            if answer is not None:
                answer = answer.replace('\n', '%%%').replace('"', '``')
            line = [
                f'"{suite_name}"',
                f'"{question}"',
                run_idx,
                # info['calls'],
                info['time'],
                info['missing'],
                info['correct'],
                f'"{answer}"'
            ]
            line = [str(x) for x in line]
            line = ','.join(line) + '\n'
            f.write(line)

    print(f'wrote `{output_file_path}`')

    # TODO: when we add the capability to run on different "systems"
    #       we can add functionality to aggregate across system
    #       and system + question


def run_suite(
        suite: Dict,
        n_runs: int,
        eval_funcs_dict: Dict,
        collection: chromadb.Collection,
        llm_from_seed: Callable,
        tools_dict: Dict
        ) -> Dict:
    """run a test suite and return results"""

    results = {}

    suite_name = suite['name']

    print(suite_name)

    success = []

    for test in tqdm.tqdm(suite['tests']):
        for run_idx in range(n_runs):
            tool = tools_dict[test['tool']]
            question = test['question']
            eval_params = dict(test['eval_func'])
            eval_func = eval_funcs_dict[eval_params['typ']]
            del eval_params['typ']

            llm = llm_from_seed(run_idx)

            start_time = time.time()

            try:
                answer = tool(question, collection, llm)
            except Exception as e:
                answer = None

            end_time = time.time()
            total_time = end_time - start_time

            # TODO: would be nice to have LLM call count here

            info = {
                'time': round(total_time, 3),
                'answer': answer,
            }

            if answer is not None:
                res_test = eval_func(answer, **eval_params)
                info['missing'] = False
                info['correct'] = res_test
                success.append(res_test)
            else:
                info['missing'] = True
                info['correct'] = False
                success.append(False)

            uid = (suite_name, question, run_idx)
            results[uid] = info

    print(sum(success), '/', len(success), 'passed')
    print('~~~~ ~~~~ ~~~~ ~~~~')

    return results


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
