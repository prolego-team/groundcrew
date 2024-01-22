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
from groundcrew.llm.openaiapi import SystemMessage, UserMessage, AssistantMessage


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
            self.messages.append(UserMessage(user_prompt))

            spinner = yaspin(text='Thinking...', color='green')
            spinner.start()
            response = self.dispatch(user_prompt)
            self.messages.append(AssistantMessage(response))
            spinner.stop()

            # TODO - handle params that should be there but are not

            self.print(response, 'agent')

    def dispatch(self, user_prompt: str) -> str:
        """
        Analyze the user's input and and either respond or choose an appropriate
        tool for generating a response. When a tool is called, the output from
        the tool will be returned as the response.

        Args:
            user_prompt (str): The user's input or question.

        Returns:
            str: The response from the tool or LLM.
        """

        dispatch_prompt = '### Tools ###\n'
        for tool in self.tools.values():
            dispatch_prompt += tool.to_yaml() + '\n\n'

        dispatch_prompt += '### Question ###\n'
        dispatch_prompt += user_prompt + '\n\n'

        # Put instructions at the end of the prompt
        dispatch_prompt += sp.CHOOSE_TOOL_PROMPT

        dispatch_messages = self.messages + [UserMessage(dispatch_prompt)]
        dispatch_response = self.llm(dispatch_messages)

        if self.config.debug:
            print(Colors.MAGENTA)
            print(dispatch_prompt, Colors.ENDC, '\n')
            print(Colors.GREEN)
            print(dispatch_response, Colors.ENDC)

        parsed_response = autils.parse_response(
            dispatch_response,
            keywords=['Response', 'Reason', 'Tool', 'Tool query']
        )

        if 'Tool' in parsed_response:
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

            if self.config.debug:
                print(f'Please standby while I run the tool {tool.name}...')
                print(f'("{parsed_response["Tool query"]}", {tool_args})')
                print()
            response = utils.highlight_code(
                tool.obj(parsed_response['Tool query'], **tool_args),
                self.config.colorscheme
            )

        else:
            response = utils.highlight_code(
                dispatch_response,
                self.config.colorscheme
            )

        return response

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
