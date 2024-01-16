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

with open('openai_key_file', 'r') as f:
    openai_key = f.read().strip()

client = oai.get_openaiai_client(openai_key)


embedding_model = oai.get_embedding_model("text-embedding-ada-002", client)
e = embedding_model(['This is a test', 'This is only a test'])
print(type(e), len(e), type(e[0][0]))


def dictionary(word):
    if word=='hacktophant':
        return 'A hacker that is also a sycophant.'
    elif word=='plasopyrus':
        return 'A hard substance found on the inner lining of a platypus bill.'
    else:
        return 'I do not know!'


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

model = 'gpt-4-1106-preview'
chat = oai.start_chat(model, client)

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
print(response)

if response.tool_calls is not None:
    tool_output_messages = [
        oai.ToolMessage(
            str(tool_functions[tool.function_name](**tool.function_args)),
            tool.tool_call_id)
        for tool in response.tool_calls
    ]
    messages += tool_output_messages

response = chat(messages, tools=tools)
print(response)
"""

from typing import Callable, Iterable
from dataclasses import dataclass
import json

import openai


@dataclass(frozen=True)
class ToolCall:
    tool_call_id: str
    tool_type: str
    function_name: str
    function_args: dict


@dataclass(frozen=True)
class SystemMessage:
    content: str
    role: str = 'system'


@dataclass(frozen=True)
class UserMessage:
    content: str
    role: str = 'user'


@dataclass(frozen=True)
class AssistantMessage:
    content: str
    tool_calls: list[ToolCall] | None
    role: str = 'assistant'


@dataclass(frozen=True)
class ToolMessage:
    content: str | None
    tool_call_id: str
    role: str = 'tool'


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


def get_openaiai_client(api_key: str | None = None) -> openai.Client:
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
    for key, value in vars(message).items():
        if value is None:
            continue

        # Handle lists of tools calls in messages
        if key == 'tool_calls' and value is not None:
            output_dict[key] = [toolcall_to_dict(tool_call) for tool_call in value]
        else:
            output_dict[key] = value

    return output_dict


def dict_to_message(message_dict: dict) -> Message:
    """Convert a dict to a message."""
    if message_dict['role'] == 'system':
        return SystemMessage(message_dict['content'])
    elif message_dict['role'] == 'user':
        return UserMessage(message_dict['content'])
    elif message_dict['role'] == 'assistant':
        if 'tool_calls' in message_dict:
            tool_calls = [
                ToolCall(
                    tool_call['id'],
                    tool_call['type'],
                    tool_call['function']['name'],
                    json.loads(tool_call['function']['arguments'])
                )
                for tool_call in message_dict['tool_calls']
            ]
        else:
            tool_calls = None
        return AssistantMessage(
            message_dict['content'],
            tool_calls
        )
    elif message_dict['role'] == 'tool':
        return ToolMessage(message_dict['content'], message_dict['tool_call_id'])
    else:
        raise ValueError('Unknown message role: ' + message_dict['role'])


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


def start_chat(model: str, client: openai.Client) -> Callable:
    """Make an LLM interface function that you can use with Messages.

    This will return a function that can be called with a list of messages.
    Optional arguments to this function should conform with parameter requirements
    of the OpenAI API, e.g., `tools`, `temperature`, `seed`, etc."""

    def chat_func(messages: list[Message], *args, **kwargs) -> Message:
        assert len(messages) > 0
        assert isinstance(messages[0], SystemMessage) or isinstance(messages[0], UserMessage)
        assert isinstance(messages[-1], UserMessage) or isinstance(messages[-1], ToolMessage)

        input_messages = [message_to_dict(message) for message in messages]
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


def get_embedding_model(model: str, client: openai.Client) -> Callable:
    """Make an embedding function."""

    def embedding_func(texts: Iterable[str]) -> list[list[str]]:
        raw_embeddings = client.embeddings.create(
            input=texts,
            model=model
        ).data[:]

        return [x.embedding for x in raw_embeddings]

    return embedding_func
