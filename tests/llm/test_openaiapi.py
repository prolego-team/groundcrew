"""
Tests for OpenAI API.
"""

from unittest.mock import patch

import pytest
import openai as oai
import json

from groundcrew.llm import openaiapi


@pytest.fixture
def openai_chat_response():
    def _chat_output(content, role):
        return oai.types.chat.ChatCompletion(
            choices=[
                oai.types.chat.chat_completion.Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=oai.types.chat.ChatCompletionMessage(
                        content=content,
                        role=role
                    )
                )
            ],
            created=1689623190,
            id="chatcmpl-xyz",
            model="gpt-3.5-turbo-0613-mock",
            system_fingerprint="fp_44709d6fcb",
            object="chat.completion",
            usage=oai.types.CompletionUsage(
                completion_tokens=99,
                prompt_tokens=99,
                total_tokens=99
            )
        )

    return _chat_output


@pytest.fixture
def openai_tool_response():
    def _chat_output(content, role, function_name, function_args):
        return oai.types.chat.ChatCompletion(
            choices=[
                oai.types.chat.chat_completion.Choice(
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                    message=oai.types.chat.ChatCompletionMessage(
                        content=content,
                        role=role,
                        tool_calls=[
                            oai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall(
                                id='some_string',
                                function=oai.types.chat.chat_completion_message_tool_call.Function(
                                    arguments=function_args,
                                    name=function_name
                                ),
                                type='function'
                            )
                        ]
                    )
                )
            ],
            created=1689623190,
            id="chatcmpl-xyz",
            model="gpt-3.5-turbo-0613-mock",
            system_fingerprint="fp_44709d6fcb",
            object="chat.completion",
            usage=oai.types.CompletionUsage(
                completion_tokens=99,
                prompt_tokens=99,
                total_tokens=99
            )
        )

    return _chat_output


# @patch('neosophia.llmtools.openaiapi.embeddings')
# def test_embeddings_tensor(embeddings_mock):
#     """test `embeddings_tensor` and `extract_embeddings` functions"""
#     n_texts = 3

#     # OpenAI API returns a data struction which contains an embedding
#     # for each input
#     example_data = {
#         'data': [
#             {
#                 'embedding': [0.0] * openaiapi.EMBEDDING_DIM_DEFAULT
#             }
#         ] * n_texts

#     }
#     embeddings_mock.return_value = example_data
#     texts = ['baloney'] * n_texts

#     res = openaiapi.embeddings_tensor(texts)
#     assert res.shape == (n_texts, openaiapi.EMBEDDING_DIM_DEFAULT)


def test_toolcall_to_dict():
    toolcall = openaiapi.ToolCall(
        'tcid',
        'function',
        'func_name',
        {
            'arg1': 42,
            'arg2': 'forty two'
        }
    )
    toolcall_dict = openaiapi.toolcall_to_dict(toolcall)
    assert toolcall_dict['id']=='tcid'
    assert toolcall_dict['type']=='function'
    assert toolcall_dict['function']=={
        'name': 'func_name',
        'arguments': '{"arg1": 42, "arg2": "forty two"}'
    }


def test_message_to_dict():
    message = openaiapi.UserMessage('This is a test.')
    message_dict = openaiapi.message_to_dict(message)
    assert message_dict['role']=='user'
    assert message_dict['content']=='This is a test.'

    message = openaiapi.AssistantMessage('This is a test.', None)
    message_dict = openaiapi.message_to_dict(message)
    assert message_dict['role']=='assistant'
    assert message_dict['content']=='This is a test.'
    assert 'tool_calls' not in message_dict

    message = openaiapi.AssistantMessage(
        'This is a test.',
        [openaiapi.ToolCall('tcid', 'function', 'func_name', {'arg1': 42})]
    )
    message_dict = openaiapi.message_to_dict(message)
    assert message_dict['role']=='assistant'
    assert message_dict['content']=='This is a test.'
    assert message_dict['tool_calls']==[{
        'id': 'tcid',
        'type': 'function',
        'function': {
            'name': 'func_name',
            'arguments': '{"arg1": 42}'
        }
    }]


@patch('groundcrew.llm.openaiapi.oai.resources.chat.completions.Completions.create')
def test_chat_completion(chat_mock, openai_chat_response, openai_tool_response):
    target_output_content = 'Who is there?'
    target_output_role = 'assistant'
    chat_mock.return_value = openai_chat_response(
        target_output_content,
        target_output_role
    )
    model = openaiapi.start_chat('test_model', 'fake_api_key')
    messages = [
        openaiapi.SystemMessage('You are a helpful assistant.'),
        openaiapi.UserMessage('Knock knock.')
    ]
    response = model(messages)
    assert response.content==target_output_content
    assert response.role==target_output_role


    target_output_function_name = 'test_func'
    target_output_function_args = '{"arg1": 42}'
    chat_mock.return_value = openai_tool_response(
        target_output_content,
        target_output_role,
        target_output_function_name,
        target_output_function_args
    )
    model = openaiapi.start_chat('test_model', 'fake_api_key')
    messages = [
        openaiapi.SystemMessage('You are a helpful assistant.'),
        openaiapi.UserMessage('Knock knock.')
    ]
    response = model(messages)
    assert response.content==target_output_content
    assert response.role==target_output_role
    assert response.tool_calls[0].function_name==target_output_function_name
    assert response.tool_calls[0].function_args==json.loads(target_output_function_args)
