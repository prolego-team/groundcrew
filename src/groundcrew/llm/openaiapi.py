"""
Wrappers for OpenAI API.
"""

from typing import Any, List, Callable, Optional
import os
import sys
from dataclasses import dataclass

import openai as oai
import torch


EMBEDDING_DIM_DEFAULT = 1536
EMBEDDING_MODEL_DEFAULT = 'text-embedding-ada-002'


@dataclass
class ToolFunction:
    name: str
    arguments: str


@dataclass
class ToolCall:
    tool_call_id: str
    tool_type: str
    function: ToolFunction


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


def get_models_list() -> List:
    """Return a list of available models."""
    model_obj = oai.models.list()
    return [model.id for model in model_obj.data]


def embeddings(client: oai.Client, texts: List[str]) -> Any:
    """get embeddings, specifying the default model"""
    return client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL_DEFAULT
    )


def chat_completion(client: oai.Client, prompt: str, model: str) -> str:
    """simple chat completion"""
    res = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    return res['choices'][0]['message']['content']


def extract_embeddings(data: Any) -> torch.Tensor:
    """extract embedings from an API response"""
    embs = [
        torch.tensor(x['embedding'])
        for x in data['data']]
    return torch.stack(embs, dim=0)


def embeddings_tensor(client: oai.Client, texts: List[str]) -> torch.Tensor:
    """get a tensor of embeddings from a list of strings"""
    embs = embeddings(client, texts)
    res = extract_embeddings(embs)
    assert res.shape == (len(texts), EMBEDDING_DIM_DEFAULT)
    return res


def message_to_dict(message: Message) -> dict:
    """Because it's much, much faster than built in dataclasses.asdict."""
    return {key:value for key,value in vars(message).items() if value is not None}


def message_from_api_response(response: dict) -> AssistantMessage:
    """Parse the API response."""
    completion = response.choices[0].message

    role = completion.role
    assert role=='assistant'

    content = completion.content
    tool_calls = completion.tool_calls
    if tool_calls is not None:
        tool_calls = [
            ToolCall(
                tool_call['id'],
                tool_call['type'],
                ToolFunction(**tool_call['function']))
            for tool_call in tool_calls
        ]

    return AssistantMessage(content, tool_calls)


def message_from_tool_call(tool_call_id: str, function_output: Any) -> ToolMessage:
    """Prepare a message from the output of a function."""
    return ToolMessage(str(function_output), tool_call_id)


def start_chat(model: str, api_key: str) -> Callable:
    """Make an LLM interface function that you can use with Messages."""

    client = get_openaiai_client(api_key=api_key)

    def chat_func(messages: List[Message], *args, **kwargs) -> Message:
        input_messages = [message_to_dict(message) for message in messages]
        try:
            response = client.chat.completions.create(
                messages=input_messages,
                model=model,
                *args,
                **kwargs
            )
            print('from api', response)
            return message_from_api_response(response)
        except oai.APIError:
            return UserMessage('There was an API error.  Please try again.')

    return chat_func
