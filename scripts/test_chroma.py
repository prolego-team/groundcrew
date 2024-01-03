"""
"""
import os
import ast
import pickle
import readline

import yaml

from tqdm import tqdm

import chromadb
import astunparse

from openai import OpenAI
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()

opj = os.path.join


def extract_functions_from_file(filepath):

    function_texts = {}

    with open(filepath, 'r') as f:
        file_lines = f.readlines()
        file_text = ''.join(file_lines)

    for node in ast.walk(ast.parse(file_text)):
        if isinstance(node, ast.FunctionDef):
            function_texts[node.name] = {
                'text': ''.join(
                    file_lines[node.lineno - 1:node.end_lineno]
                ),
                'start_line': node.lineno,
                'end_line': node.end_lineno
            }

    return function_texts


def build_llm_client():
    client = OpenAI()

    def chat_complete(prompt):
        complete = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return complete.choices[0].message.content

    return chat_complete


def codebase_qa(question, collection, chat_complete):

    out = collection.query(
        query_texts=[question],
        n_results=10,
        where={'type': 'function'}
    )

    prompt = 'Answer the question given the following data. Be descriptive in your answer and provide full filepaths and line numbers, but do not provide code.'
    prompt += f'Question {question}\n'
    ids = out['ids'][0]
    metadatas = out['metadatas'][0]
    documents = out['documents'][0]

    for id_, metadata, doc in zip(ids, metadatas, documents):
        prompt += f'ID:\n'
        prompt += f'{id_}\n\n'
        #prompt += f'metadata:\n'
        #prompt += f'{metadata}\n\n'
        prompt += 'start line: ' + str(metadata['start_line']) + '\n'
        prompt += 'end line: ' + str(metadata['end_line']) + '\n'
        prompt += f'description:\n'
        prompt += f'{doc}\n'
        prompt += '--------\n'

    #print('PROMPT', prompt, '\n\n')

    return chat_complete(prompt)


def main():

    chat_complete = build_llm_client()

    with open('config.yaml', 'r') as f:
        data = yaml.safe_load(f)

    repo_directory = data['repository']
    db_path = data['db_path']

    client = chromadb.PersistentClient(db_path)

    python_files = []
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            if file.endswith('.py'):
                if 'llmtools' not in opj(root, file):
                    continue
                if 'test_' in file:
                    continue
                filepath = opj(root, file).split(repo_directory)[1][1:]
                python_files.append(filepath)

    c = 0
    if not os.path.exists('function_descriptions.pkl'):

        function_descriptions = {}
        for file in tqdm(python_files):
            filepath = opj(repo_directory, file)
            file_functions = extract_functions_from_file(filepath)
            for function_name, function_info in file_functions.items():
                function_text = function_info['text']
                function_id = file + '::' + function_name

                prompt = 'Generate a human readable description for the following Python function.\n'
                prompt += 'Function ID: ' + function_id
                prompt += 'Function Text:\n' + function_text
                function_descriptions[function_id] = f'Function_id: {function_id}\n'
                function_descriptions[function_id] += chat_complete(prompt)

                c += 1
                if c > 10:
                    break

            if c > 10:
                break

        with open('function_descriptions.pkl', 'wb') as f:
            pickle.dump(function_descriptions, f)
    else:
        with open('function_descriptions.pkl', 'rb') as f:
            function_descriptions = pickle.load(f)

    print('FUNCTIONS')
    print(function_descriptions.keys())
    print('\n')

    collection = client.get_or_create_collection(
        name='database', embedding_function=default_ef
    )

    documents = []
    metadatas = []
    ids = []

    dp = []
    for file in python_files:

        filepath = opj(repo_directory, file)

        file_functions = extract_functions_from_file(filepath)
        for function_name, function_info in file_functions.items():

            function_text = function_info['text']
            start_line = function_info['start_line']
            end_line = function_info['end_line']

            function_id = file + '::' + function_name

            function_description = function_descriptions.get(function_id)
            if function_description is None:
                continue

            metadata = {
                'type': 'function',
                'name': function_name,
                'id': function_id,
                'filepath': file,
                'function_text': function_text,
                'start_line': start_line,
                'end_line': end_line
            }

            ids.append(function_id)
            metadatas.append(metadata)
            documents.append(function_description)

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    while True:

        prompt = input('> ')
        if not prompt:
            prompt = 'What is the name of the function that finds pdfs in a directory?'

        response = codebase_qa(prompt, collection, chat_complete)

        print(response)
        print('\n', 80 * '*', '\n')


if __name__ == '__main__':
    main()

