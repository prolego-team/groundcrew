"""
Evaluation framework.
"""

from typing import Dict, List, Callable
import datetime
import os
import re
import time
import sys

import chromadb
import click
import git
import pickle
import tqdm
import yaml

from groundcrew.dataclasses import Config
from groundcrew import agent
from groundcrew import constants
from groundcrew import evaluation as ev
from groundcrew import utils
from groundcrew import tools

# TODO: find a better place for this
#       Maybe in the eval config?
REPO_COMMIT_HASH = '5b5b2c6d16e94edae7a03fac2a666b95713c1904'


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

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir_path = f'{output_dir_prefix}_{timestamp}'
    os.makedirs(output_dir_path, exist_ok=False)

    output_file_prefix = 'eval'

    # ~~~~ load a single system (for now) that we will evaluate

    system_default = load_system_from_config(
        config_file_path=config,
        llm_model_name=model,
        hash_check=REPO_COMMIT_HASH
    )

    # ~~~~ load the evaluation suites that we will run

    evals = load_evals(evals)

    # ~~~~ build a set of evaluation functions

    llm = utils.build_llm_completion_client(model)

    eval_funcs = dict(
        match_word_any=match_word_any,
        contains_all=contains_all,
        always_pass=always_pass,
        eval_with_llm=build_eval_with_llm(llm)
    )

    # ~~~~ run some checks before starting to make sure our evals are valid

    for suite in evals:
        ev.verify_suite(suite, system_default, eval_funcs)

    # ~~~~ run evaluations

    # for now, we are only testing on one "system"

    results = {}

    for suite in evals:
        results_suite = run_suite(
            system=system_default,
            suite=suite,
            n_runs=n_runs,
            eval_funcs=eval_funcs,
        )
        results = {**results, **results_suite}

    # ~~~~ save result pickle

    output_file_path = os.path.join(output_dir_path, f'{output_file_prefix}.pkl')
    with open(output_file_path, 'wb') as f:
        pickle.dump(results, f)

    # ~~~~ save result CSV

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
        system: ev.System,
        suite: ev.EvalSuite,
        n_runs: int,
        eval_funcs: Dict
        ) -> Dict:
    """Run a test suite on a system and return results"""

    results = {}

    print(suite.name)

    success = []

    for test in tqdm.tqdm(suite.tests):
        for run_idx in range(n_runs):
            build_tool = system.tools[test.tool]
            tool_params = test.params
            eval_params = dict(test.eval_func)
            eval_func = eval_funcs[eval_params['type']]
            del eval_params['type']

            try:
                # special case for testing the Agent which also needs a chat LLM
                llm = system.llm_from_seed(run_idx)
                if test.tool == 'Agent':
                    chat_llm = system.chat_llm_from_seed(run_idx)
                    tool = build_tool(llm, chat_llm)
                else:
                    tool = build_tool(llm)

                start_time = time.time()

                answer = tool(**tool_params)
            except Exception as e:
                import traceback
                print('exception while running tool:', e)
                print(traceback.format_exc())
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

            uid = (suite.name, test.name, run_idx)
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


def contains_all(text: str, checks: List[str]) -> bool:
    """
    Check if text contains ALL checks.
    """

    for check in checks:
        if check not in text:
            return False
    return True


def always_pass(text: str) -> bool:
    """
    Always pass.
    """
    return True


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


def load_system_from_config(
        config_file_path: str,
        llm_model_name: str,
        hash_check: str
        ) -> ev.System:
    """
    Default system loading behavior, contstructs a system from the same config
    file used by run
    """

    # ~~~~ load config ~~~~

    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    config = Config(**config)

    # ~~~~ check the code version of the repo we are going to test

    repo = git.Repo(os.path.expanduser(config.repository))
    assert repo.head.commit.hexsha == hash_check

    # ~~~~ build the system (collection, llm, etc)

    # For now we'll assume that `run.py` has been run
    # and the database is present

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
    llm_from_seed = lambda _: utils.build_llm_completion_client(llm_model_name)
    chat_llm_from_seed = lambda _: utils.build_llm_chat_client(llm_model_name)

    tools_filepath = os.path.join(config.cache_dir, 'tools.yaml')
    tool_descs = utils.setup_and_load_yaml(tools_filepath, 'tools')

    def agent_wrapper(llm: Callable, chat_llm: Callable) -> Callable:
        """
        Given a chat llm, build a function that can interact
        with an agent.
        """

        # TODO: this is not ideal. eventually I want to rewrite / refactor `setup_tools`
        #       to avoid evaluation potentially updating the cache
        tools_for_agent = utils.setup_tools(
            modules_list=config.Tools,
            tool_descriptions=tool_descs,
            collection=collection,
            llm=llm,
            working_dir_path=os.path.expanduser(config.repository)
        )

        def run_agent(user_prompts: List[str]) -> str:
            """run the agent"""
            agent_obj = agent.Agent(config, collection, chat_llm, tools_for_agent)
            res = ''
            for prompt in user_prompts:
                print('>>>> INPUT:', prompt)
                res = agent_obj.interact_functional(prompt)
                print('>>>> OUTPUT:', res)
            print('>>>> run_agent DONE')

            return res
        return run_agent

    # This is actually a dictionary of tool constructors, adapted
    # to take an LLM. This is because during testing need the ability
    # to construct a new LLM with a different seed for each iteration.

    tools_dict = {
        'CodebaseQATool': lambda x: tools.CodebaseQATool(
            base_prompt=tool_descs['CodebaseQATool']['base_prompt'],
            collection=collection,
            llm=x
        ),
        'SingleDocstringTool': lambda x: tools.SingleDocstringTool(
            base_prompt=tool_descs['SingleDocstringTool']['base_prompt'],
            collection=collection,
            llm=x
        ),
        'LintFileTool': lambda x: tools.LintFileTool(
            base_prompt=tool_descs['LintFileTool']['base_prompt'],
            collection=collection,
            llm=x,
            working_dir_path=os.path.expanduser(config.repository)
        ),
        'Agent': agent_wrapper
    }

    return ev.System(
        tools=tools_dict,
        llm_from_seed=llm_from_seed,
        chat_llm_from_seed=chat_llm_from_seed
    )


def load_evals(evals_file_path: str) -> list[ev.EvalSuite]:
    """Load a list of evaluation suites to run"""

    with open(evals_file_path, 'r') as f:
        evals = yaml.safe_load(f)

    try:
        evals = [ev.parse_suite(x) for x in evals]
    except Exception as e:
        print('Error parsing evaluation suits:')
        print(e)
        sys.exit()

    return evals


if __name__ == '__main__':
    main()
