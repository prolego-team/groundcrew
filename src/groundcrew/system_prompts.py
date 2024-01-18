"""
"""

DOCSTRING_PROMPT = """Your response must be formatted such that the first line is the function definition, and below it is the docstring. Do not engage in conversation or print any of the function's code. Your response must include ```python your response. If there are multiple functions, separate them by two newlines.
"""

SUMMARIZE_FILE_PROMPT = """
Your task is to generate a concise summary of the above text and describe what the file is for. Keep your summary to 5 sentences or less.

"""

SUMMARIZE_CODE_PROMPT = """
Your task is to generate a concise summary of the above Python code. Keep your summary to 5 sentences or less. Include in your summary:
    - Dependencies
    - Important functions and clasess
    - Relevant information from comments and docstrings
"""

CHOOSE_TOOL_PROMPT = """Your task is to choose the correct tool and parameters to answer the question. Each tool has a list of parameters along with which are required or not. If a parameter is required, you MUST generate a value for it. Do not engage in any conversation - your answer must be in the following format.

Reason: Describe your reasoning for why this tool was chosen in 3 sentences or less.
Tool: Tool Name
Parameter_0: Parameter_0 Name | Variable_0 value | parameter type
Parameter_1: Parameter_1 Name | Variable_1 value | parameter type
...
Parameter_N: Parameter_N Name | Variable_N value | parameter type
"""

TOOL_GPT_PROMPT = """Your task is to take as input a Python `Tool` class and create a description of the `__call__` method in YAML format like the example below. All `Tools` will include a `user_prompt` parameter in the `__call__` method.

Instructions:
- Your output should be formatted exactly like the example
- Your output should be in correct YAML format.
- The `base_prompt` you generate should instruct the Tool on what its task is
- `user_prompt` should be excluded when generating the description.

Restrictions:
- Do not include ```yaml in your answer
- Do not engage in any conversation.
- Do not include ```yaml in your answer
- The description should talk about the class as a `Tool` and not mention it being a Python class
- Do not include anything that isn't valid YAML in your answer
- Do not include backticks in your answer
- Do not include ```yaml in your answer

### Example Input ###
class ToolExample(Tool):

    def __init__(self, base_prompt: str, collection, llm):
        """ """
        super().__init__(base_prompt, collection, llm)

    def __call__(self, user_prompt: str, parameter_1: str):

        # Logic here with parameter_1
        output = ... # output from database from parameter_1

        full_prompt = output + '\n' + self.base_prompt + user_prompt
        return self.llm(full_prompt)

### Example Output ###
  - name: ToolExample
    description: This tool takes a user_prompt and parameter_1, does some logic with parameter_1, then uses a large language model to answer the user's question.
    base_prompt: Your task is to answer the question given the following data. Be descriptive in your answer.
    params:
      parameter_1:
        description: Description of parameter_1
        type: str
        required: true
"""
