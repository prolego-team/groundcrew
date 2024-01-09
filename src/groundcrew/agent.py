"""
"""
from typing import Any

from groundcrew import agent_utils as autils, system_prompts as sp
from groundcrew.tools import CodebaseQATool


class Agent:

    def __init__(
            self,
            config,
            collection,
            llm,
            tools):

        self.config = config
        self.collection = collection
        self.llm = llm
        self.tools = tools

    def run(self):

        while True:

            user_prompt = input('> ')
            if not user_prompt:
                user_prompt = 'What is the name of the function that finds pdfs in a directory?'

            tool, args = self.choose_tool(user_prompt)
            response = tool.obj(user_prompt, **args)

            print(response)


    def choose_tool(self, user_prompt):

        base_prompt = sp.CHOOSE_TOOL_PROMPT
        base_prompt += '### Question ###\n'
        base_prompt += user_prompt + '\n\n'

        base_prompt += '### Tools ###\n'
        for tool in self.tools.values():
            base_prompt += tool.to_yaml() + '\n\n'

        tool_prompt = base_prompt

        tool = None
        while tool is None:

            # Choose a Tool
            tool_response = self.llm(tool_prompt)
            parsed_tool_response = autils.parse_response(
                tool_response, keywords=['Reason', 'Tool'])

            if parsed_tool_response['Tool'] in self.tools:
                tool = self.tools[parsed_tool_response['Tool']]

            else:
                # TODO
                tool_name = parsed_tool_response['Tool']
                error_msg = f'{tool_name} not in TOOLS. '
                error_msg += 'Choose a tool from the TOOLS section'
                step = {
                    'status': 'error',
                    'message': error_msg
                }
                tool_failure_steps.append(step)
                self.all_steps.append(step)
                self.log.add(
                    'TOOL AGENT', f'ERROR - Wrong Tool chosen: {tool_name}')
                self.log.add('TOOL AGENT', error_msg)

            return tool, self.extract_params(parsed_tool_response)

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
                        param_value=None

                args[param_name.replace(' ', '')] = param_value

        #if 'kwargs' in args:
        #    args = self.substitute_variable_values_kwargs(args)

        return args
