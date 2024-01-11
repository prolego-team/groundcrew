"""
Tools.
"""

from typing import Callable

import chromadb

from groundcrew.dataclasses import Chunk


def format_chunk(chunk: Chunk, include_text: bool) -> str:
    """
    Format a chunk object as a string for use in a prompt.
    """

    # TODO - change Document based on chunk type
    # TODO - start and end lines won't be needed if typ is a file

    prompt = f'Name: {chunk.name}\n'
    prompt += f'Type: {chunk.typ}\n'
    if include_text:
        prompt += f'Full Text: {chunk.text}\n'
    prompt += f'Document: {chunk.document}\n'
    prompt += f'Start Line: {chunk.start_line}\n'
    prompt += f'End Line: {chunk.end_line}\n'
    return prompt


def codebase_qa(prompt: str, collection: chromadb.Collection, llm: Callable) -> str:
    """
    Ask a question about the codebase and get an answer
    """

    out = collection.query(
        query_texts=[prompt],
        n_results=5,
        where={'type': 'function'}
    )

    ids = out['ids'][0]
    metadatas = out['metadatas'][0]
    documents = out['documents'][0]

    # TODO - update typ
    chunks = []
    for id_, metadata, doc in zip(ids, metadatas, documents):
        chunks.append(
            Chunk(
                name=metadata['name'],
                uid=metadata['id'],
                typ='function',
                text=metadata['function_text'], # TODO - change to just text
                document=doc,
                filepath=metadata['filepath'],
                start_line=metadata['start_line'],
                end_line=metadata['end_line']
            )
        )

    base_prompt = 'Answer the question given the following data. Be descriptive in your answer and provide full filepaths and line numbers, but do not provide code.\n'
    prompt = base_prompt + f'Question: {prompt}\n\n'

    for chunk in chunks:
        prompt += format_chunk(chunk, include_text=False)
        prompt += '--------\n\n'

    return llm(prompt)





