"""
"""
import chromadb

from chromadb.utils import embedding_functions
from groundcrew.tools import codebase_qa


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

        while True:

            base_prompt = 'Choose the correct tool to answer the following question.'

            prompt = input('> ')
            if not prompt:
                prompt = 'What is the name of the function that finds pdfs in a directory?'

            response = codebase_qa(prompt, self.collection, self.llm)

            print(response)
            print('\n', 80 * '*', '\n')

