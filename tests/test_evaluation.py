"""
Tests for evaluation functions.
"""

import dataclasses

from groundcrew import evaluation

import pytest


def test_parse_verify():
    """Test parsing and verifying eval suites."""

    # ~~~~ parse

    correct = dict(
        name='Tests',
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

    # bad tool

    bad_suite = dataclasses.replace(
        correct_suite,
        tests=[
            evaluation.EvalTest(
                name='Test A', tool='GarbgeTool', params=dict(x=1, y=2), eval_func=dict()
            )
        ]
    )
    with pytest.raises(AssertionError) as ae:
        evaluation.verify_suite(bad_suite, system, eval_funcs)
    assert str(ae.value) == '`Tests`:`Test A`: tool from suite not in system'

    # tool parameter mismatch
    bad_suite.tests[0] = dataclasses.replace(bad_suite.tests[0], tool='BaloneyTool')
    with pytest.raises(AssertionError) as ae:
        evaluation.verify_suite(bad_suite, system, eval_funcs)
    assert str(ae.value) == '`Tests`:`Test A`: tool parameter mismatch'

    # bad eval functions
    bad_suite.tests[0] = dataclasses.replace(
        bad_suite.tests[0],
        params=dict(user_prompt='Hello, world!', a=6, b=6),
    )
    with pytest.raises(AssertionError) as ae:
        evaluation.verify_suite(bad_suite, system, eval_funcs)
    assert str(ae.value) == '`Tests`:`Test A`: eval_func missing type'

    bad_suite.tests[0] = dataclasses.replace(
        bad_suite.tests[0],
        eval_func=dict(type='is_correct')
    )
    with pytest.raises(AssertionError) as ae:
        evaluation.verify_suite(bad_suite, system, eval_funcs)
    assert str(ae.value) == '`Tests`:`Test A`: eval_func type `is_correct` not in eval_funcs'

    bad_suite.tests[0] = dataclasses.replace(
        bad_suite.tests[0],
        eval_func=dict(type='contains')
    )
    with pytest.raises(AssertionError) as ae:
        evaluation.verify_suite(bad_suite, system, eval_funcs)
    assert str(ae.value) == '`Tests`:`Test A`: eval_func parameter mismatch'
