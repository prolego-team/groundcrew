"""
"""
import os
import ast
import types
import importlib

from typing import Any, Callable, Dict, List, Tuple

import yaml

import astunparse

from openai import OpenAI

from groundcrew import system_prompts as sp
from groundcrew.dataclasses import Tool


def build_llm_client(model='gpt-4-1106-preview'):
    """

    """
    if 'gpt' in model:
        client = OpenAI()

        def chat_complete(prompt):
            complete = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return complete.choices[0].message.content

    return chat_complete


def setup_and_load_yaml(filepath: str, key: str) -> Dict[str, Dict[str, Any]]:
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
        modules_list: List[Dict[str, Any]],
        tool_descriptions: Dict[str, Dict[str, str]]) -> Dict[str, Tool]:
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

    tools = {}
    for module_dict in modules_list:
        module_name = module_dict['module']
        module = importlib.import_module(module_name)
        tools_list = module_dict['tools']

        tool_dict = build_tool_dict_from_module(
            module, tools_list)

        for func_name, (callable_func, func_str) in tool_dict.items():
            if func_name in tool_descriptions:
                print(f'Loading description for {func_name}')
                description_yaml = yaml.dump(
                    tool_descriptions[func_name], sort_keys=False)
            else:
                print(f'Generating description for {func_name}')
                tool_yaml = convert_function_str_to_yaml(func_str)
                tool_yaml = tool_yaml.replace('**kwargs', 'kwargs')
                tool_dict = yaml.safe_load(tool_yaml)[0]
                description_yaml = yaml.dump(tool_dict, sort_keys=False)

            tool = Tool(
                name=func_name,
                function_str=func_str,
                description=description_yaml,
                call=callable_func
            )
            tools[func_name] = tool

    return tools


def build_tool_dict_from_module(
        module: types.ModuleType,
        tool_names: List[str]) -> Dict[str, Tuple[Callable, str]]:
    """
    Takes a list of python modules as input and builds a dictionary containing
    the function name as the key and a tuple containing the callable function
    and its string representation as the value.

    Args:
        module (types.ModuleType): The module containing the functions.
        function_names (List[str]): A list of function names to include in the
        dictionary.

    Returns:
        function_dict (Dict[str, Tuple[Callable, str]]): A dictionary mapping
        function names to tuples containing the callable function and its
        string representation.
    """
    tool_dict = {}
    with open(module.__file__, 'r') as f:
        file_text = ''.join(f.readlines())

    for node in ast.walk(ast.parse(file_text)):
        if isinstance(node, ast.ClassDef) and node.name in tool_names:
            print('node:', node)
            print('name:', node.name)
            class_instance = getattr(module, node.name)
            callable_function = class_instance
            exit()
            function_dict[node.name] = (
                callable_function, astunparse.unparse(node))

    return function_dict


def convert_function_str_to_yaml(function_str: str) -> str:
    """
    Convert a given function string to YAML format using a GPT-4 model.

    Args:
        function_str (str): The string representation of a function.

    Returns:
        str: The YAML representation of the given function string.
    """
    prompt = sp.FUNCTION_GPT_PROMPT + '\n\n' + function_str
    llm = build_llm_client()
    return llm(prompt)


def save_tools_to_yaml(tools: Dict[str, Tool], filename: str) -> None:
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
        tool_yaml = yaml.safe_load(tool.description)
        tools_list.append(tool_yaml)

    # Wrap the list in a dictionary with the key 'tools'
    data = {'tools': tools_list}

    # Write the data to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
