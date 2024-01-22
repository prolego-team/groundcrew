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

    import pickle

    # look at descriptions in cache and fix them!

    descriptions_file = os.path.join(config.cache_dir, 'descriptions.pkl')
    with open(descriptions_file, 'rb') as f:
        descriptions = pickle.load(f)

    print(descriptions.keys())

    descriptions_fixed = {}
    fixed_count = 0
    for k, v in descriptions.items():
        if k.startswith(config.repository):
            k_fixed = os.path.relpath(k, config.repository)
            print(k, '->', k_fixed)
            assert k == (config.repository + '/' + k_fixed)
            fixed_count += 1
            k = k_fixed
        descriptions_fixed[k] = v

    print('fixed keys of', fixed_count, '/', len(descriptions_fixed), 'descriptions')

    assert len(set(descriptions.keys())) == len(set(descriptions_fixed.keys()))

    if fixed_count > 0:
        with open(descriptions_file, 'wb') as f:
            pickle.dump(descriptions_fixed, f)


if __name__ == '__main__':
    main()
