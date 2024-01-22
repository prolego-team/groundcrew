"""
Verify the database with different tests.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import sys

import click

import chromadb
import click
import git
import tqdm
import yaml

from groundcrew.dataclasses import Config
from groundcrew import code
from groundcrew import constants
from groundcrew import utils
from groundcrew import tools


@click.command()
@click.option('--config', '-c', default='config.yaml')
def main(config: str):
    """main program"""

    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    config = Config(**config)

    # get files in repo
    file_paths = list(code.get_committed_files(config.repository, config.extensions))
    file_paths = [x.split(os.path.abspath(config.repository))[1][1:] for x in file_paths]


    # client and collection
    client = chromadb.PersistentClient(config.db_path)
    collection = client.get_or_create_collection(
        name='database',
        embedding_function=constants.DEFAULT_EF
    )

    # load the whole collection
    # this might not scale, but it works for now
    collection_dict = collection.get(
        include=['metadatas', 'documents']
    )

    docs = {
        k: v for k, v
        in zip(collection_dict['ids'], collection_dict['documents'])}

    metadatas = {
        k: v for k, v
        in zip(collection_dict['ids'], collection_dict['metadatas'])
    }

    print('total ids:', len(docs))
    empty_docs = []
    for uid, doc in docs.items():
        print(uid)
        if not doc:
            print('empty document')
            empty_docs.append(uid)
    print('empty docs:', len(empty_docs))
    print('\t', empty_docs)

    # look for files that are missing from database
    file_ids = set([
        x for x in docs
        if ':' not in x
    ])

    # right now the file_ids are still full paths unfortunately
    file_paths = [os.path.join(config.repository, x) for x in file_paths]

    print('db file ids:    ', len(file_ids))
    print('repo file paths:', len(file_paths))
    print('matches:', len(set(file_ids).intersection(set(file_paths))))


if __name__ == '__main__':
    main()
