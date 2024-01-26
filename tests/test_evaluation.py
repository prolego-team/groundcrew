"""
Tests for evaluation functions.
"""


from groundcrew import evaluation

import pytest


def test_parse_verify():
    """Test parsing and verifying eval suites."""

    # ~~~~ parse

    correct = dict(
        name='Valid Tests',
        tests=[
            dict(
                name='Test A',
                tool='BaloneyTool',
                params=dict(user_prompt='Hello, world!', a=6, b=6),
                eval_func=dict(type='contains', check='yes')
            )
        ]
    )
    correct_suite = evaluation.parse_suite(correct)

    assert isinstance(correct_suite, evaluation.EvalSuite)

    malformed = dict(correct)
    malformed['tests'] = list(malformed['tests'])
    malformed['tests'].append(dict(name='Test B', tool='baloney'))

    with pytest.raises(Exception):
        evaluation.parse_suite(malformed)

    # ~~~~ verify

    system = evaluation.System(
        tools=dict(
            BaloneyTool=lambda x: (
                lambda user_prompt, a, b: 'Generic LLM response'),
            DerpTool=lambda x: (
                lambda user_prompt, c, d: 'As an AI model I am not allowed to do that.'),
            Agent=lambda x, y: (
                lambda user_prompts: 'I am feeling lazy today. Do it yourself.')
        ),
        llm_from_seed=None,
        chat_llm_from_seed=None
    )

    eval_funcs = dict(
        contains=lambda text, check: check in text
    )

    # basic correctly formatted suite
    evaluation.verify_suite(correct_suite, system, eval_funcs)

    # special case for "Agent" tool
    correct_suite.tests.append(evaluation.EvalTest(
        name='Test Agent',
        tool='Agent',
        params=dict(user_prompts=['Answer the question!', 'I said answer the question!']),
        eval_func=dict(type='contains', check='The answer is')
    ))
    evaluation.verify_suite(correct_suite, system, eval_funcs)

    # TODO: test bad tools
    # TODO: test bad eval functions
    # TODO: test bad eval function params

