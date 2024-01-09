"""
"""

CHOOSE_TOOL_PROMPT = """Your task is to choose the correct tool and parameters to answer the following question. Do not engage in any conversation. Your answer must be in the following format.

Reason: Describe your reasoning for why this tool was chosen in 3 sentences or less.
Tool: Tool Name
Parameter_0: Parameter_0 Name | Variable value | parameter type
...
Parameter_N: Parameter_N Name | Variable value | parameter type

"""

CODEBASE_QA_PROMPT = "Your task is to answer the question given the following data. Be descriptive in your answer and provide full filepaths and line numbers.\n"

TOOL_GPT_PROMPT = """Your task is to take as input a Python `Tool` class and create a description of the `__call__` method in YAML format like the example below. All `Tools` will include a `prompt` parameter in the `__call__` method.

Instructions:
- Your output should be formatted exactly like the example
- Your output should be in correct YAML format.
- The `base_prompt` you generate should instruct the Tool on what its task is
- `prompt` should be excluded when generating the description.

Restrictions:
- Do not include ```yaml in your answer
- Do not engage in any conversation.
- Do not include ```yaml in your answer
- The description should talk about the class as a `Tool` and not mention it being a Python class
- Do not include anything that isn't valid YAML in your answer
- Do not include backticks in your answer
- Do not include ```yaml in your answer

### Example Input ###
class CodebaseQATool(Tool):

    def __init__(self, base_prompt: str, collection, llm):
        """ """
        super().__init__(base_prompt, collection, llm)

    def __call__(self, prompt: str, include_code: str):

        chunks = self.query_codebase(prompt)

        prompt = self.base_prompt + '### Question ###\n'
        prompt += f'{prompt}\n\n'

        for chunk in chunks:
            prompt += code.format_chunk(chunk, include_text=include_code)
            prompt += '--------\n\n'

        print(prompt)

        return self.llm(prompt)

    def query_codebase(self, prompt: str):

        out = self.collection.query(
            query_texts=[prompt],
            n_results=5,
            where={'type': 'function'}
        )

        return [
            Chunk(
                name=metadata['name'],
                uid=metadata['id'],
                typ='function',
                text=metadata['function_text'],
                document=doc,
                filepath=metadata['filepath'],
                start_line=metadata['start_line'],
                end_line=metadata['end_line']
            ) for id_, metadata, doc in zip(
                out['ids'][0], out['metadatas'][0], out['documents'][0]
                )
        ]

### Example Output ###
  - name: CodebaseQATool
    description: This tool takes a user question as input, searches the codebase for relevant files and functions, then uses a large language model to analyze the data and answer the user's question
    base_prompt: Your task is to answer the question given the following data. Be descriptive in your answer and provide full filepaths and line numbers.
    params:
      include_code:
        description: Whether or not to include code in the LLM's response
        type: bool
        required: true
"""
