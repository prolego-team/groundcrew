"""
Main agent class interacting with a user
"""
import os
import fcntl
import select
import inspect
import readline
import subprocess

from typing import Any, Callable

from chromadb import Collection

from yaspin import yaspin
from yaspin.core import Yaspin

from groundcrew import agent_utils as autils, system_prompts as sp, utils
from groundcrew.dataclasses import Colors, Config, Tool
from groundcrew.llm.openaiapi import (AssistantMessage, Message, SystemMessage,
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
        self.messages: list[Message] = []
        self.spinner: Yaspin | None = None
        self.shell_process = None
        self.shell_mode = False

        self.shell_system_prompt = sp.SHELL_PROMPT + '\n\n'
        self.shell_system_prompt += sp.CHOOSE_TOOL_PROMPT + '\n### Tools ###\n'
        for tool in self.tools.values():
            self.shell_system_prompt += tool.to_yaml() + '\n\n'

        self.colors = {
            'system': Colors.YELLOW,
            'user': Colors.GREEN,
            'agent': Colors.BLUE
        }

    def run_shell_command(self, command: str) -> tuple[str, str]:
        """
        """
        if self.shell_process:

            # Write the command and flush immediately
            self.shell_process.stdin.write(command + '\n')
            self.shell_process.stdin.flush()

            output = ''
            error = ''

            # Set stdout and stderr to non-blocking mode
            stdout_fd = self.shell_process.stdout.fileno()
            stderr_fd = self.shell_process.stderr.fileno()
            fl_stdout = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
            fl_stderr = fcntl.fcntl(stderr_fd, fcntl.F_GETFL)
            fcntl.fcntl(stdout_fd, fcntl.F_SETFL, fl_stdout | os.O_NONBLOCK)
            fcntl.fcntl(stderr_fd, fcntl.F_SETFL, fl_stderr | os.O_NONBLOCK)

            # Keep reading until there's no more output or error
            while True:
                read_ready, _, _ = select.select(
                    [
                        self.shell_process.stdout,
                        self.shell_process.stderr
                    ], [], [], 0.1
                )

                if self.shell_process.stdout in read_ready:
                    try:
                        chunk = os.read(stdout_fd, 1024).decode()
                        if chunk:
                            output += chunk
                    except BlockingIOError:
                        # No more output available
                        pass

                if self.shell_process.stderr in read_ready:
                    try:
                        chunk = os.read(stderr_fd, 1024).decode()
                        if chunk:
                            error += chunk
                    except BlockingIOError:
                        # No more error available
                        pass

                if not read_ready:
                    # No more output or error
                    break

            return output, error
        return '', 'Error: Shell process not initialized.'

    def start_shell_session(self) -> None:
        """
        Start a new persistent shell session
        """
        if not self.shell_process:
            self.shell_process = subprocess.Popen(
                ['/bin/zsh'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self.run_shell_command(f'export CONDA_DIR={self.config.conda_dir}')
            self.run_shell_command('source src/groundcrew/.shellrc')
            self.shell_mode = True
            self.shell_messages = []

    def stop_shell_session(self) -> None:
        """
        Terminate the shell process
        """
        if self.shell_process:
            self.shell_process.terminate()
            self.shell_process = None
            self.shell_mode = False

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
        print(utils.highlight_code(text, self.config.colorscheme))

    def interact(self, user_prompt: str) -> None:
        """
        Process a user prompt and call dispatch

        Args:
            user_prompt (str): The user's input or question.
        Returns:
            None
        """

        if not self.config.debug:
            self.spinner = yaspin(text='Thinking...', color='green')
            self.spinner.start()

        self.dispatch(user_prompt)

        # Append dispatch messages except for the system prompt
        self.messages.extend(self.dispatch_messages)

        if self.config.debug:
            self.print_message_history(self.messages)
        else:
            self.spinner.stop()

        # Response with the last message from the agent
        self.print(self.messages[-1].content, 'agent')

    def get_current_conda_env(self) -> str:
        # Command to echo the CONDA_DEFAULT_ENV variable
        command = 'echo $CONDA_DEFAULT_ENV'
        stdout, _ = self.run_shell_command(command)

        # The output will include the environment name, strip any whitespace
        env_name = stdout.strip()

        # Return the environment name or a default string if not found
        return env_name if env_name else 'base'

    def run(self):
        """
        Continuously listen for user input and respond using the chosen tool
        based on the input.
        """

        current_env = 'base'

        while True:

            user_prompt = ''

            while user_prompt == '':

                str_prompt = '[user] > '
                if self.shell_mode:
                    current_env = self.get_current_conda_env()
                    str_prompt = f'[{self.config.shell} | {current_env}] > '

                user_prompt = input(str_prompt)

                if '\\code' in user_prompt:
                    code_mode = True
                    print(Colors.YELLOW)
                    print('Code mode activated — type \end to submit')
                    print(Colors.ENDC)
                    user_prompt += '\n'
                    line = input('')
                    while '\\end' not in line:
                        user_prompt += line + '\n'
                        line = input('')

                elif '\\shell' in user_prompt and not self.shell_mode:
                    self.shell_mode = True
                    print(Colors.CYAN)
                    print('Shell mode activated — type \exit to exit')
                    print('Current shell:', self.config.shell)
                    print('Current dir:', self.run_shell_command('pwd')[0])
                    print('To ask the LLM a question, start your command with \\help')
                    print(Colors.ENDC)
                    self.start_shell_session()
                    user_prompt = ''
                    continue

                elif '\\exit' in user_prompt and self.shell_mode:
                    print(Colors.CYAN)
                    print('Shell mode exited')
                    print(Colors.ENDC)
                    self.stop_shell_session()
                    user_prompt = ''
                    continue

            if self.shell_mode:

                # User is asking a question to the LLM
                if user_prompt.startswith('\\help'):
                    user_prompt = user_prompt.replace('\\help', '')

                    current_dir, _ = self.run_shell_command('pwd')

                    user_shell_messages = 'Anaconda env: ' + current_env + '\n'
                    user_shell_messages += 'Current dir: ' + current_dir + '\n\n'

                    user_shell_messages += user_prompt + '\n\n' + '\n'.join(
                        self.shell_messages) + '\n'

                    print(user_shell_messages)

                    system_prompt = sp.SHELL_PROMPT + '\n\n'
                    system_prompt += sp.CHOOSE_TOOL_PROMPT + '\n\n'

                    self.dispatch(
                        system_prompt=system_prompt,
                        user_prompt=user_shell_messages
                    )

                    # Append dispatch messages except for the system prompt
                    self.messages.extend(self.dispatch_messages)

                    self.print_message_history(self.messages)

                    # Response with the last message from the agent
                    self.print(self.messages[-1].content, 'agent')

                else:
                    std_out, std_err = self.run_shell_command(user_prompt)
                    user_message = f'Command: {user_prompt}\n\n'
                    if std_out:
                        user_message += f'Stdout:\n{std_out}\n\n'
                        print(Colors.GREEN)
                        print(std_out)
                        print(Colors.ENDC)
                    if std_err:
                        user_message += f'Stderr:\n{std_err}\n\n'
                        print(Colors.RED)
                        print(std_err)
                        print(Colors.ENDC)

                    self.shell_messages.append(user_message)

            else:
                user_prompt = user_prompt.replace('\\code', '')
                if user_prompt == 'exit' or user_prompt == 'quit' or user_prompt == 'q':
                    break

                self.interact(user_prompt)

    def run_tool(self, parsed_response: dict[str, str | list[str]]) -> str:
        """
        Runs the Tool selected by the LLM.

        Args:
            parsed_response (Dict): A dictionary containing the parsed data
            from LLM.

        Returns:
            tool_response (str): The response from the tool.
        """

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

    def interact_functional(self, user_prompt: str) -> str:
        """
        Process a user prompt and call dispatch
        Args:
            user_prompt (str): The user's input or question.
        Returns:
            the system's response
        """

        # spinner = yaspin(text='Thinking...', color='green')
        # spinner.start()

        self.dispatch(user_prompt)
        self.messages.extend(self.dispatch_messages)

        # if self.spinner is not None:
        #     self.spinner.stop()

        content = self.messages[-1].content
        self.print(content, 'agent')
        return content

    def dispatch(self, system_prompt, user_prompt: str) -> None:
        """
        Analyze the user's input and either respond or choose an appropriate
        tool for generating a response. When a tool is called, the output from
        the tool will be returned as the response.

        Args:
            user_prompt (str): The user's input or question.

        Returns:
            None
        """

        if self.spinner is not None:
            self.spinner.stop()

        #system_prompt = sp.AGENT_PROMPT + '\n\n' + sp.CHOOSE_TOOL_PROMPT + '\n\n'
        system_prompt += '### Tools ###\n'
        for tool in self.tools.values():
            system_prompt += tool.to_yaml() + '\n\n'

        # the message history involved in solving the current user_prompt
        self.dispatch_messages = []

        user_question = '\n\n### Question ###\n' + user_prompt
        self.dispatch_messages.append(UserMessage(user_question))

        while True:

            self.spinner = yaspin(text='Thinking...', color='green')
            self.spinner.start()

            # Choose tool or get a response
            select_tool_response = self.llm(
                [SystemMessage(system_prompt)] +
                self.messages +
                self.dispatch_messages
            )

            # Add response to the dispatch messages as an assistant message
            self.dispatch_messages.append(select_tool_response)
            self.spinner.stop()

            # Parse the tool selection response
            parsed_select_tool_response = autils.parse_response(
                select_tool_response.content,
                keywords=['Response', 'Reason', 'Tool', 'Tool query']
            )

            # No Tool selected - this should be an answer
            if 'Tool' not in parsed_select_tool_response:
                break

            self.spinner = yaspin(
                text='Running ' + parsed_select_tool_response['Tool'],
                color='green'
            )
            self.spinner.start()

            # Run Tool
            try:
                tool_response = self.run_tool(parsed_select_tool_response)
            except Exception:
                tool_response = ''

            self.spinner.stop()

            tool_response_message = 'Tool response\n' + tool_response
            tool_response_message += user_question + '\n\n'
            tool_response_message += sp.TOOL_RESPONSE_PROMPT
            self.dispatch_messages.append(UserMessage(tool_response_message))

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

    def print_message_history(self, messages):
        print('\n', '*' * 50, '\n')
        for message in messages:
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
