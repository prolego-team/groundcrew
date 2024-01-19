"""
Evaluation framework.

Current focus is on testing individual tools.
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
from groundcrew import code
from groundcrew import constants
from groundcrew import utils
from groundcrew import tools


@click.command()
@click.option('--config', '-c', default='config.yaml')
@click.option('--model', '-m', default='gpt-4-1106-preview')
@click.option('--evals', '-e', default='data/evaluation.yaml')
@click.option('--n_runs', '-n', default=3)
@click.option('--output_dir_prefix', '-o', default='eval')
def main(
        config: str,
        model: str,
        evals: str,
        n_runs: int,
        output_dir_prefix: str):
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
        name=constants.DEFAULT_COLLECTION_NAME,
        embedding_function=constants.DEFAULT_EF
    )

    # OpenAI models can't be created with a seed
    # so this is a simple wrapper that ignores the seed
    llm_from_seed = lambda _: utils.build_llm_client(model)
    llm = utils.build_llm_client(model)

    tools_filepath = os.path.join(config.cache_dir, 'tools.yaml')
    tool_descs = utils.setup_and_load_yaml(tools_filepath, 'tools')

    # This is actually a dictionary of tool constructors, adapted
    # to take a Collection and LLM.

    # The tuple of Collection and LLM (more or less) defines the
    # "system" that we are trying to test, since the most obvious
    # way we could get different behavior from the system is with
    # different summaries / embeddings / embedding method (Collection)
    # and different LLM for generative capabilites.

    # Tool base_prompts are passed in here for now but we could make
    # those part of the system later.

    tools_dict = {
        'CodebaseQATool': lambda c, l: tools.CodebaseQATool(
            base_prompt=tool_descs['CodebaseQATool']['base_prompt'],
            collection=c,
            llm=l
        ),
        'SingleDocstringTool': lambda c, l: tools.SingleDocstringTool(
            base_prompt=tool_descs['SingleDocstringTool']['base_prompt'],
            collection=c,
            llm=l
        ),
        'LintFileTool': lambda c, l: tools.LintFileTool(
            base_prompt=tool_descs['LintFileTool']['base_prompt'],
            collection=c,
            llm=l,
            working_dir_path=config.repository
        )
    }

    # build a set of evaluation functions

    eval_funcs_dict = dict(
        match_word_any=match_word_any,
        eval_with_llm=build_eval_with_llm(llm)
    )

    # ~~~~ run evaluations

    # For now, we are only testing on one "system"
    # (one combination of Collection / LLM)

    results = {}

    for suite in evals:
        results_suite = run_suite(
            suite=suite,
            n_runs=n_runs,
            eval_funcs_dict=eval_funcs_dict,
            tools_dict=tools_dict,

            collection=collection,
            llm_from_seed=llm_from_seed,
        )
        results = {**results, **results_suite}

    # ~~~~ save result CSV

    output_file_prefix = 'eval'
    output_file_path = os.path.join(output_dir_path, f'{output_file_prefix}.csv')
    # header = 'suite,question,run,calls,time,missing,correct,answer\n'
    header = 'suite,test,run,time,missing,correct,answer\n'
    with open(output_file_path, 'w') as f:
        f.write(header)
        for (suite_name, test_name, run_idx), info in results.items():
            answer = info['answer']
            if answer is not None:
                answer = answer.replace('\n', '%%%').replace('"', '``')
            line = [
                f'"{suite_name}"',
                f'"{test_name}"',
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
    #       and system + test


def run_suite(
        suite: Dict,
        n_runs: int,
        eval_funcs_dict: Dict,
        tools_dict: Dict,

        collection: chromadb.Collection,
        llm_from_seed: Callable,
        ) -> Dict:
    """run a test suite and return results"""

    results = {}

    suite_name = suite['name']

    print(suite_name)

    success = []

    for test in tqdm.tqdm(suite['tests']):
        for run_idx in range(n_runs):
            test_name = test['name']
            build_tool = tools_dict[test['tool']]
            tool_params = test['params']
            eval_params = dict(test['eval_func'])
            eval_func = eval_funcs_dict[eval_params['typ']]
            del eval_params['typ']

            llm = llm_from_seed(run_idx)

            start_time = time.time()

            try:
                tool = build_tool(collection, llm)
                answer = tool(**tool_params)
            except Exception as e:
                print('exception while running tool:', e)
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

            uid = (suite_name, test_name, run_idx)
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


def build_eval_with_llm(llm: Callable) -> Callable:
    """
    Build an evaluation function that uses an LLM to perform
    potentially more complex evaluations.
    """
    def eval_with_llm(text: str, instructions: str) -> bool:
        """
        Evaluate a result using and LLM and instructions.
        """

        prompt = (
            f'### Text ###\n{text}\n\n' +
            f'### Evaluation Instructions ###\n{instructions}\n\n' +
            'Your task is to evaluate the above Text given the Instructions. ' +
            'If the text satisfies the instructions, answer "yes" otherwise answer "no".'
        )

        res = llm(prompt)
        res = res.lower()
        if 'yes' in res:
            return True
        return False

    return eval_with_llm


if __name__ == '__main__':
    main()
