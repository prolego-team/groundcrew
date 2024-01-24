"""
Tools for evaluation.
"""

from typing import Any, Callable
from dataclasses import dataclass
import inspect


@dataclass(frozen=True)
class System:
    """
    A sytem to test.
    The combination of a set of tools and and LLM determines the system.
    Individual tools hold the chromadb collection so it doesn't need
    to be part of this.
    """
    # collection: chromadb.Collection
    tools: dict[str, Callable]
    llm_from_seed: Callable
    chat_llm_from_seed: Callable


@dataclass(frozen=True)
class EvalTest:
    """A single test to perform on a tool."""
    name: str
    tool: str
    params: dict[str, str]
    eval_func: dict[str, str]


@dataclass(frozen=True)
class EvalSuite:
    """Named collection of tests."""
    name: str
    tests: list[EvalTest]


def parse_suite(eval_dict: dict[str, Any]) -> EvalSuite:
    """Parse a a dict into an EvalSuite."""

    return EvalSuite(
        name=eval_dict['name'],
        tests=[
            EvalTest(**x) for x in eval_dict['tests']
        ]
    )


def verify_suite(suite: EvalSuite, system: System, eval_funcs: dict[str, Callable]) -> None:
    """perform checks on an eval suite"""
    for test in suite.tests:
        uid = f'`{suite.name}`:`{test.name}`'
        assert test.tool in system.tools, f'{uid}: tool from suite not in system'

        # instantiate the tool to check params
        if test.tool == 'Agent':
            tool = system.tools[test.tool](None, None)
        else:
            tool = system.tools[test.tool](None)

        actual_params = inspect.signature(tool).parameters
        compare_params = set(test.params.keys()) == set(actual_params.keys())

        assert compare_params, f'{uid}: tool parameter mismatch'

        assert 'type' in test.eval_func, f'{uid}: eval_func missing type'
        ef_typ = test.eval_func['type']
        assert ef_typ in eval_funcs, f'{uid}: eval_func type `{ef_typ}` not in eval_funcs'

        params = dict(test.eval_func)
        del params['type']
        params['text'] = None

        actual_params = inspect.signature(eval_funcs[ef_typ]).parameters
        compare_params = set(params.keys()) == set(actual_params.keys())

        assert compare_params, f'{uid}: eval_func parameter mismatch'
