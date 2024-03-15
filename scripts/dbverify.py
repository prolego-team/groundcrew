"""
Verify the database with different tests.
"""

import os
import pickle

import chromadb
import click
import yaml

from groundcrew.dataclasses import Config
from groundcrew import code
from groundcrew import constants


@click.command()
@click.option('--config', '-c', default='config.yaml')
def main(config: str):
    """main program"""

    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    config = Config(**config)

    # get files in repo
    file_paths = list(code.get_committed_files(os.path.expanduser(config.repository), config.extensions))
    file_paths = [x.split(os.path.abspath(os.path.expanduser(config.repository)))[1][1:] for x in file_paths]

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

    # ~~~~ find empty documents

    empty_docs = []
    for uid, doc in docs.items():
        if not doc:
            print('empty document')
            empty_docs.append(uid)

    print('empty docs:', len(empty_docs))
    print('\t', empty_docs)
    print()

    # ~~~~ find counts of documents by type

    docs_by_type = {}
    for uid, metadata in metadatas.items():
        doc_type = metadata['type']
        ids = docs_by_type.setdefault(doc_type, [])
        ids.append(uid)

    print('counts by type:')
    for k, v in docs_by_type.items():
        print('\t', k, len(v))
    print()

    # ~~~~ find files that are missing from database

    file_ids = set([
        x for x in docs
        if ':' not in x
    ])

    print('db file ids:    ', len(file_ids))
    print('repo file paths:', len(file_paths))
    print('matches:', len(set(file_ids).intersection(set(file_paths))))

    # ~~~~ fix descriptions of in cache that don't use the relative path

    descriptions_file = os.path.join(config.cache_dir, 'descriptions.pkl')
    with open(descriptions_file, 'rb') as f:
        descriptions = pickle.load(f)

    descriptions_fixed = {}
    fixed_count = 0
    for k, v in descriptions.items():
        if k.startswith(os.path.expanduser(config.repository)):
            k_fixed = os.path.relpath(k, os.path.expanduser(config.repository))
            print(k, '->', k_fixed)
            assert k == (os.path.expanduser(config.repository) + '/' + k_fixed)
            fixed_count += 1
            k = k_fixed
        descriptions_fixed[k] = v

    print('fixed keys of', fixed_count, '/', len(descriptions_fixed), 'descriptions')

    assert len(set(descriptions.keys())) == len(set(descriptions_fixed.keys()))

    if fixed_count > 0:
        with open(descriptions_file, 'wb') as f:
            pickle.dump(descriptions_fixed, f)
        print(f'updated {descriptions_file}')


if __name__ == '__main__':
    main()
