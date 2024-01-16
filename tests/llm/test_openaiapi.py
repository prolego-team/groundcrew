"""
Tests for OpenAI API.
"""

from unittest.mock import patch

import pytest
import openai
import json

from groundcrew.llm import openaiapi


@pytest.fixture
def openai_chat_response():
    def _chat_output(content, role):
        return openai.types.chat.ChatCompletion(
            choices=[
                openai.types.chat.chat_completion.Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=openai.types.chat.ChatCompletionMessage(
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
            usage=openai.types.CompletionUsage(
                completion_tokens=99,
                prompt_tokens=99,
                total_tokens=99
            )
        )

    return _chat_output


@pytest.fixture
def openai_tool_response():
    def _chat_output(content, role, function_name, function_args):
        return openai.types.chat.ChatCompletion(
            choices=[
                openai.types.chat.chat_completion.Choice(
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                    message=openai.types.chat.ChatCompletionMessage(
                        content=content,
                        role=role,
                        tool_calls=[
                            openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall(
                                id='some_string',
                                function=openai.types.chat.chat_completion_message_tool_call.Function(
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
            usage=openai.types.CompletionUsage(
                completion_tokens=99,
                prompt_tokens=99,
                total_tokens=99
            )
        )

    return _chat_output

@pytest.fixture
def openai_embedding_response():
    def _embedding_response(num_embeddings, embedding_dim):
        return openai.types.CreateEmbeddingResponse(
            data=[
                openai.types.Embedding(
                    embedding=[0.42]*embedding_dim,
                    index=i,
                    object='embedding'
                )
                for i in range(num_embeddings)
            ],
            model='fake_embedding_model',
            object='list',
            usage=openai.types.create_embedding_response.Usage(
                prompt_tokens=99,
                total_tokens=99
            )
        )

    return _embedding_response


@patch('groundcrew.llm.openaiapi.openai.resources.embeddings.Embeddings.create')
def test_embeddings(embeddings_mock, openai_embedding_response):
    """test embedding functions"""
    n_texts = 3
    embedding_dim = 10

    embeddings_mock.return_value = openai_embedding_response(n_texts, embedding_dim)
    texts = ['baloney'] * n_texts

    client = openaiapi.get_openaiai_client('fake_api_key')
    embedding_func = openaiapi.get_embedding_model('fake_model', client)
    results = embedding_func(texts)

    assert len(results)==len(texts)
    assert len(results[0])==embedding_dim
    assert isinstance(results[0][0], float)


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
    assert toolcall_dict=={
        'id': 'tcid',
        'type': 'function',
        'function': {
            'name': 'func_name',
            'arguments': '{"arg1": 42, "arg2": "forty two"}'
        }
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


@patch('groundcrew.llm.openaiapi.openai.resources.chat.completions.Completions.create')
def test_chat_completion(chat_mock, openai_chat_response, openai_tool_response):
    target_output_content = 'Who is there?'
    target_output_role = 'assistant'
    chat_mock.return_value = openai_chat_response(
        target_output_content,
        target_output_role
    )
    client = openaiapi.get_openaiai_client('fake_api_key')
    model = openaiapi.start_chat('test_model', client)
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
    model = openaiapi.start_chat('test_model', client)
    messages = [
        openaiapi.SystemMessage('You are a helpful assistant.'),
        openaiapi.UserMessage('Knock knock.')
    ]
    response = model(messages)
    assert response==openaiapi.AssistantMessage(
        target_output_content,
        [openaiapi.ToolCall(
            'some_string',
            'function',
            target_output_function_name,
            json.loads(target_output_function_args))
        ]
    )
