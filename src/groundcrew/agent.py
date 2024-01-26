"""
Main agent class interacting with a user
"""
import inspect
import readline

from typing import Any, Callable

from yaspin import yaspin
from chromadb import Collection

from groundcrew import agent_utils as autils, system_prompts as sp, utils
from groundcrew.dataclasses import Colors, Config, Tool
from groundcrew.llm.openaiapi import (AssistantMessage, SystemMessage,
                                      UserMessage)


class Agent:
    """
    A class representing an agent that interacts with a user to execute various
    tools based on user prompts.

    Attributes:
        config (dict): Configuration settings for the agent.
        collection (object): The collection or database the agent interacts with.
        chat_llm (object): A chat-based LLM used by the agent for processing
        and interpreting prompts.
        tools (dict): A dictionary of tools available for the agent to use.

    Methods:
        run(): Continuously process user inputs and execute corresponding tools.
        dispatch(user_prompt): Analyze the user prompt and select the
        appropriate tool for response or respond directly if appropriate.
    """
    def __init__(
            self,
            config: Config,
            collection: Collection,
            chat_llm: Callable,
            tools: dict[str, Tool]):
        """
        Constructor
        """
        self.config = config
        self.collection = collection
        self.llm = chat_llm
        self.tools = tools
        self.messages = [SystemMessage(sp.AGENT_PROMPT)]

        self.colors = {
            'system': Colors.YELLOW,
            'user': Colors.GREEN,
            'agent': Colors.BLUE
        }

    def print(self, text: str, role: str) -> None:
        """
        Helper function to print text with a given color and role.

        Args:
            text (str): The text to print.
            role (str): The role of the text to print.

        Returns:
            None
        """
        print(self.colors[role])
        print(f'[{role}]')
        print(Colors.ENDC)
        print(text)

    def interact(self, user_prompt: str) -> None:
        """
        Process a user prompt and call dispatch

        Args:
            user_prompt (str): The user's input or question.
        Returns:
            None
        """
        self.messages.append(UserMessage(user_prompt))
        if not self.config.debug:
            spinner = yaspin(text='Thinking...', color='green')
            spinner.start()
        response = self.dispatch(user_prompt)
        self.messages.append(AssistantMessage(response))
        if not self.config.debug:
            spinner.stop()
        self.print(response, 'agent')

    def run(self):
        """
        Continuously listen for user input and respond using the chosen tool
        based on the input.
        """
        while True:

            user_prompt = ''

            while user_prompt == '':
                user_prompt = input('[user] > ')
                if '\\code' in user_prompt:
                    print(Colors.YELLOW)
                    print('Code mode activated — type \end to submit')
                    print(Colors.ENDC)
                    user_prompt += '\n'

                    line = input('')

                    while '\\end' not in line:
                        user_prompt += line + '\n'
                        line = input('')

            user_prompt = user_prompt.replace('\\code', '')

            self.interact(user_prompt)

    def run_tool(self, parsed_response):

        tool_selection = parsed_response['Tool']
        if tool_selection not in self.tools:
            return 'The LLM tried to call a function that does not exist.'

        tool = self.tools[tool_selection]
        tool_args = self.extract_params(parsed_response)

        expected_tool_args = inspect.signature(tool.obj).parameters

        # Filter out incorrect parameters
        new_args = {}
        for param_name, val in tool_args.items():
            if param_name in expected_tool_args:
                new_args[param_name] = val
        tool_args = new_args

        # Add any missing parameters - default to None for now.
        # In the future we'll probably want the LLM to regenerate params
        for param_name in expected_tool_args.keys():
            if param_name == 'user_prompt':
                continue
            if param_name not in tool_args:
                tool_args[param_name] = None

        if self.config.debug:
            print(f'Please standby while I run the tool {tool.name}...')
            print(f'("{parsed_response["Tool query"]}", {tool_args})')
            print()

        return tool.obj(parsed_response['Tool query'], **tool_args)

        return utils.highlight_code(
            tool.obj(parsed_response['Tool query'], **tool_args),
            self.config.colorscheme
        )

    def dispatch(self, user_prompt: str) -> str:
        """
        """

        # TODO - update parser to return one Tool
        # Update language for tool response
        # Merge dispatch_messages and messages

        system_prompt = sp.CHOOSE_TOOL_PROMPT + '\n\n'
        system_prompt += '### Tools ###\n'
        for tool in self.tools.values():
            system_prompt += tool.to_yaml() + '\n\n'

        self.dispatch_messages = [SystemMessage(system_prompt)]

        user_question = '\n\n### Question ###\n' + user_prompt
        self.dispatch_messages.append(
            UserMessage(user_question)
        )

        loop_count = 0
        while True:

            # Choose tool or get a response
            select_tool_response = self.llm(self.dispatch_messages)
            self.dispatch_messages.append(select_tool_response)

            # Parse the tool selection response
            parsed_select_tool_response = autils.parse_response(
                select_tool_response.content,
                keywords=['Response', 'Reason', 'Tool', 'Tool query']
            )

            # No Tool selected - this should be an answer or conversation of sorts
            if 'Tool' not in parsed_select_tool_response:
                return utils.highlight_code(
                    select_tool_response.content,
                    self.config.colorscheme
                )

            # Run Tool
            tool_response = self.run_tool(parsed_select_tool_response)

            self.dispatch_messages.append(
                UserMessage('Tool response\n' + tool_response + user_question + '\n\nIf you can answer the complete question do so, otherwise choose a Tool.')
            )

            self.print_dm()

            print('End of loop', loop_count)
            loop_count += 1

        return dispatch_response

    def extract_params(
            self,
            parsed_data: dict[str, str | list[str]]) -> dict[str, Any]:
        """
        Extract parameters from LLM response

        Args:
        parsed_data (Dict): A dictionary containing the parsed data from LLM.

        Returns:
            tool (Tool): The tool extracted from available tools based on
            'Tool' key in parsed_data.
            args (Dict): A dictionary of arguments to be passed to the
            function.
        """

        param_prefix = 'Parameter_'

        # Create a dictionary of arguments to be passed to the function
        args = {}
        for key, value in parsed_data.items():
            if key.startswith(param_prefix):

                # Value does not contain name | value | type
                if len(value) != 3:
                    continue

                param_name = value[0]
                param_value = value[1]
                param_type = value[2].strip(' ')

                # Cast the values if needed
                if param_type == 'int':
                    param_value = int(param_value)
                elif param_type == 'float':
                    param_value = float(param_value)
                elif param_type == 'bool':
                    if param_value.lower() == 'true':
                        param_value = True
                    elif param_value.lower() == 'false':
                        param_value = False
                    else:
                        param_value = None

                args[param_name.replace(' ', '')] = param_value

        return args

    def run_with_prompts(self, prompts: list[str]):
        """
        Process a list of user prompts and respond using the chosen tool
        based on the input.

        Args:
            prompts (List[str]): List of prompts to be processed by the agent.
        """
        for i, user_prompt in enumerate(prompts):

            if i == 0:
                self.print(user_prompt, 'user')

            self.interact(user_prompt)

            if i < len(prompts) - 1:
                print('Next prompt:')
                self.print(prompts[i + 1], 'user')
                input('\nPress enter to continue...\n')

        self.run()

    def print_dm(self):
        print('\n', '*' * 50, '\n')
        for message in self.dispatch_messages:
            if message.role == 'user':
                color = Colors.GREEN
            elif message.role == 'system':
                color = Colors.RED
            elif message.role == 'assistant':
                color = Colors.BLUE

            print('Role:', message.role)
            print(color)
            print(message.content)
            print(Colors.ENDC)
        print('\n', '*' * 50, '\n')
        input('End of printing, press enter to continue...')
