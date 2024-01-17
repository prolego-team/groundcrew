"""
"""
import os
import ast
import inspect
import importlib

from typing import Any, Callable

import yaml
import astunparse

from openai import OpenAI
from chromadb import Collection

from groundcrew import system_prompts as sp
from groundcrew.dataclasses import Tool
from groundcrew.llm import openaiapi


def build_llm_chat_client(model: str = 'gpt-4-1106-preview') -> Callable[[str], str]:
    """

    """
    if 'gpt' in model:
        client = openaiapi.get_openaiai_client()
        chat_session = openaiapi.start_chat(model, client)

        def chat(messages: list[dict[str, str]]) -> str:
            messages_openai = [openaiapi.dict_to_message(message) for message in messages]
            response = chat_session(messages_openai)
            messages.append(response)
            return openaiapi.message_to_dict(response)

    return chat


def build_llm_completion_client(model: str = 'gpt-4-1106-preview') -> Callable[[str], str]:
    """

    """
    if 'gpt' in model:
        client = openaiapi.get_openaiai_client()
        completion = openaiapi.start_chat(model, client)

        def chat_complete(prompt):
            messages = [
                openaiapi.SystemMessage("You are a helpful assistant."),
                openaiapi.UserMessage(prompt)
            ]
            response = completion(messages)
            return response.content

    return chat_complete


def setup_and_load_yaml(filepath: str, key: str) -> dict[str, dict[str, Any]]:
    """
    Helper function to set up workspace, create a file if it doesn't exist,
    and load data from a YAML file.

    Args:
        filepath (str): The path to the YAML file.
        key (str): The key to extract data from the loaded YAML file.

    Returns:
        dict: A dictionary containing processed data extracted from the YAML
        file. If the file doesn't exist or the data is None, an empty
        dictionary is returned.
    """

    # Create a file if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass

    # Load data from the file
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # Process the loaded data
    if data is None:
        return {}

    return {item['name']: item for item in data[key]}


def setup_tools(
        modules_list: list[dict[str, Any]],
        tool_descriptions: dict[str, dict[str, str]],
        collection: Collection,
        llm: Callable) -> dict[str, Tool]:
    """
    This function sets up tools by generating a dictionary of Tool objects
    based on the given modules and tool descriptions.

    Args:
        modules_list (list): A list of dictionaries containing modules and the
        functions to be used from those modules.
        tool_descriptions (dict): A dictionary containing descriptions for
        specific tools.

    Returns:
        tools (dict): A dictionary containing Tools with each key being the
        tool name
    """

    # Parameters available to a tool constructor
    params = {
        'collection': collection,
        'llm': llm
    }

    tools = {}
    for module_dict in modules_list:
        module_name = module_dict['module']
        module = importlib.import_module(module_name)
        tools_list = module_dict['tools']

        # Loop through tools and generate descriptions
        with open(module.__file__, 'r') as f:
            file_text = ''.join(f.readlines())

        for node in ast.walk(ast.parse(file_text)):
            if isinstance(node, ast.ClassDef) and node.name in tools_list:

                # Actual code of the class
                tool_code = astunparse.unparse(node)

                # Description already generated and in yaml file, so load it.
                if node.name in tool_descriptions:
                    print(f'Loading description for {node.name}...')
                    tool_yaml = yaml.dump(
                        tool_descriptions[node.name], sort_keys=False)

                else:
                    print(f'Generating description for {node.name}...')

                    # Generate description of the Tool in YAML format
                    tool_yaml = convert_tool_str_to_yaml(tool_code, llm)

                    # In case the LLM put ```yaml at the beginning and and ```
                    # at the end
                    if '```yaml' in tool_yaml:
                        tool_yaml = '\n'.join(tool_yaml[1:-1])

                # Convert YAML to a dictionary
                tool_info_dict = yaml.safe_load(tool_yaml)
                if isinstance(tool_info_dict, list):
                    tool_info_dict = tool_info_dict[0]

                params['base_prompt'] = tool_info_dict['base_prompt']

                tool_constructor = getattr(module, node.name)
                tool_params = inspect.signature(tool_constructor).parameters

                args = {}
                for param_name, value in params.items():
                    if param_name in tool_params:
                        args[param_name] = value

                # Create an instance of a tool object
                tool_obj = tool_constructor(**args)

                # Check that the tool object has the correct signature
                assert 'prompt' in inspect.signature(tool_obj).parameters, 'Tool must have a prompt parameter'

                assert inspect.signature(tool_obj).return_annotation == str, 'Tool must return a string'

                # Add the tool to the tools dictionary
                tools[node.name] = Tool(
                    name=node.name,
                    code=tool_code,
                    description=tool_info_dict['description'],
                    base_prompt=tool_info_dict['base_prompt'],
                    params=tool_info_dict['params'],
                    obj=tool_obj)

    return tools


def convert_tool_str_to_yaml(function_str: str, llm: Callable) -> str:
    """
    Convert a given tool string to YAML format using a GPT-4 model.

    Args:
        function_str (str): The string representation of a function.

    Returns:
        str: The YAML representation of the given function string.
    """
    prompt = sp.TOOL_GPT_PROMPT + '\n\n' + function_str
    return llm(prompt)


def save_tools_to_yaml(tools: dict[str, Tool], filename: str) -> None:
    """
    Converts a dictionary of tools into YAML format and saves it to a file.

    Args:
        tools (dict): A dictionary containing Tools with each key being the
        tool name
        filename (str): The name of the file to save the YAML data to.

    Returns:
        None
    """

    # Convert the tools dictionary into a list of dictionaries
    tools_list = []
    for tool in tools.values():
        tool_dict = {}
        tool_dict['name'] = tool.name
        tool_dict['description'] = tool.description
        tool_dict['base_prompt'] = tool.base_prompt
        tool_dict['params'] = tool.params
        tools_list.append(tool_dict)

    # Wrap the list in a dictionary with the key 'tools'
    data = {'tools': tools_list}

    # Write the data to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    print(f'Saved {filename}\n')
