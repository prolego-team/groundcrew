"""
Wrappers for OpenAI API.

The following data structures very closely mirror those required by the
OpenAI API.
- `ToolCall` encapsulates tool calling parameters.  These are created by the
  assistant/LLM.
- `ToolMessage` is a message representing the response from a tool call.
- `{User|System|Assistant}Message` objects are messages with associated rules
  and contents.  `AssistantMessage` may also hold `ToolCall`s.

Example Usage:

import groundcrew.llm.openaiapi as oai

# Embeddings
embedding_model = oai.get_embedding_model("text-embedding-ada-002", openai_key)
embeddings = embedding_model(['This is a test', 'This is only a test'])

# Chat
def dictionary(word):
    ...

function_descriptions = [
    {
        'description': 'Lookup a word in the dictionary.',
        'name': 'dictionary',
        'parameters': {
            'type': 'object',
            'properties': {
                'word': {
                    'type': 'string',
                    'description': 'The word to look up.'
                },
            },
            'required': ['word']
        }
    }
]
tools = [{'type':'function', 'function':func} for func in function_descriptions]
tool_functions = {'dictionary': dictionary}

chat = oai.start_chat('gpt-4-1106-preview', openai_key)

messages = [
    oai.SystemMessage(
        'You are a helpful assistant that finds the meaning of novel words not '
        'frequently encountered in the English language.'
    ),
    oai.UserMessage('What do the words hacktophant and plasopyrus mean?')
]
response = chat(
    messages,
    tools=tools
)
messages.append(response)

if response.tool_calls is not None:
    tool_output_messages = [
        oai.message_from_tool_call(
            tool.tool_call_id,
            tool_functions[tool.function_name](**tool.function_args)
        )
        for tool in response.tool_calls
    ]
    messages += tool_output_messages

final_response = chat(messages, tools=tools)
"""

from typing import Any, Callable, Iterable
from dataclasses import dataclass
import json

import openai


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


def get_openaiai_client(api_key: str) -> openai.Client:
    """Get an OpenAI API client."""
    return openai.OpenAI(api_key=api_key)


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
        except openai.APIError:
            return UserMessage('There was an API error.  Please try again.')

    return chat_func


def get_embedding_model(model: str, api_key: str) -> Callable:
    """Make an embedding function."""

    client = get_openaiai_client(api_key=api_key)

    def embedding_func(texts: Iterable[str]) -> list[list[str]]:
        raw_embeddings = client.embeddings.create(
            input=texts,
            model=model
        ).data[:]

        return [x.embedding for x in raw_embeddings]

    return embedding_func
