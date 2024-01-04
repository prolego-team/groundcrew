"""
Wrappers for OpenAI API.

The following data structures very closely mirror those required by the
OpenAI API.
- `ToolCall` encapsulates tool calling parameters.  These are created by the
  assistant/LLM.
- `ToolMessage` is a message representing the response from a tool call.
- `{User|System|Assistant}Message` objects are messages with associated rules
  and contents.  `AssistantMessage` may also hold `ToolCall`s.
"""

from typing import Any, Callable
import os
import sys
from dataclasses import dataclass
import json

import openai as oai
import torch


EMBEDDING_DIM_DEFAULT = 1536
EMBEDDING_MODEL_DEFAULT = 'text-embedding-ada-002'


@dataclass
class ToolCall:
    tool_call_id: str
    tool_type: str
    function_name: str
    function_args: dict


@dataclass
class SystemMessage:
    content: str
    role: str = 'system'


@dataclass
class UserMessage:
    content: str
    role: str = 'user'


@dataclass
class AssistantMessage:
    content: str
    tool_calls: list[ToolCall] | None
    role: str = 'assistant'


@dataclass
class ToolMessage:
    content: str | None
    tool_call_id: str
    role: str = 'tool'


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


def get_openaiai_client(api_key: str) -> oai.Client:
    """Get an OpenAI API client."""
    return oai.OpenAI(api_key=api_key)


def load_api_key(file_path: str) -> str:
    """load the API key from a text file"""
    if not os.path.isfile(file_path):
        print(f'OpenAI API key file `{file_path}` not found!')
        sys.exit()
    with open(file_path, 'r') as txt_file:
        res = txt_file.read().rstrip()
    if not res:
        print(f'Key file `{file_path}` empty!')
        sys.exit()

    return res


def embeddings(client: oai.Client, texts: list[str]) -> Any:
    """get embeddings, specifying the default model"""
    return client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL_DEFAULT
    )


def extract_embeddings(data: Any) -> torch.Tensor:
    """extract embedings from an API response"""
    embs = [
        torch.tensor(x['embedding'])
        for x in data['data']]
    return torch.stack(embs, dim=0)


def embeddings_tensor(client: oai.Client, texts: list[str]) -> torch.Tensor:
    """get a tensor of embeddings from a list of strings"""
    embs = embeddings(client, texts)
    res = extract_embeddings(embs)
    assert res.shape == (len(texts), EMBEDDING_DIM_DEFAULT)
    return res


def toolcall_to_dict(tool_call: ToolCall) -> dict:
    """Convert a ToolCall to a dict that can be embedded in an API message."""
    return {
        'id': tool_call.tool_call_id,
        'type': 'function',
        'function': {
            'name': tool_call.function_name,
            'arguments': json.dumps(tool_call.function_args)
        }
    }

def message_to_dict(message: Message) -> dict:
    """Convert a message to a dict that can be passed to the API.

    This is much, much faster than the built in dataclasses.asdict function."""

    output_dict = {}
    for key,value in vars(message).items():
        if value is None:
            continue

        # Handle lists of tools calls in messages
        if isinstance(value, list):
            output_dict[key] = [
                vi if not isinstance(vi, ToolCall) else toolcall_to_dict(vi)
                for vi in value
            ]
        else:
            output_dict[key] = value

    return output_dict


def message_from_api_response(response: dict) -> AssistantMessage:
    """Parse the API response."""
    completion = response.choices[0].message

    if completion.tool_calls is not None:
        tool_calls = [
            ToolCall(
                tool_call.id,
                tool_call.type,
                tool_call.function.name,
                json.loads(tool_call.function.arguments))
            for tool_call in completion.tool_calls
        ]
    else:
        tool_calls = None

    return AssistantMessage(completion.content, tool_calls)


def message_from_tool_call(tool_call_id: str, function_output: Any) -> ToolMessage:
    """Prepare a message from the output of a function."""
    return ToolMessage(str(function_output), tool_call_id)


def start_chat(model: str, api_key: str) -> Callable:
    """Make an LLM interface function that you can use with Messages.

    This will return a function that can be called with a list of messages.
    Optional arguments to this function should conform with parameter requirements
    of the OpenAI API, e.g., `tools`, `temperature`, `seed`, etc."""

    client = get_openaiai_client(api_key=api_key)

    def chat_func(messages: list[Message], *args, **kwargs) -> Message:
        input_messages = [message_to_dict(message) for message in messages]
        print(input_messages)
        try:
            response = client.chat.completions.create(
                messages=input_messages,
                model=model,
                *args,
                **kwargs
            )
            return message_from_api_response(response)
        except oai.APIError:
            return UserMessage('There was an API error.  Please try again.')

    return chat_func
