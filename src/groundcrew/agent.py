"""
"""
from groundcrew import system_prompts as sp
from groundcrew.tools import CodebaseQATool


class Agent:

    def __init__(
            self,
            config,
            collection,
            llm):

        self.config = config
        self.collection = collection
        self.llm = llm

    def run(self):

        codebase_qa_tool = CodebaseQATool(
            sp.CODEBASE_QA_PROMPT, self.collection, self.llm)

        while True:

            base_prompt = 'Choose the correct tool to answer the following question.'

            prompt = input('> ')
            if not prompt:
                prompt = 'What is the name of the function that finds pdfs in a directory?'

            #response = codebase_qa(prompt, self.collection, self.llm, False)

            response = codebase_qa_tool(prompt, include_code=True)

            print(response)
            print('\n', 80 * '*', '\n')

