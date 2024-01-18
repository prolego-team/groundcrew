"""
Figure out how chromadb embedding stuff works.
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Ben Zimmer


import chromadb
import numpy as np

from groundcrew import constants


def main():
    """main program"""

    # can replace with another text document
    with open('mtgrules.txt', 'r') as f:
        rules = f.readlines()
    rules = ''.join(rules)

    client = chromadb.PersistentClient('baloney')

    collection = client.get_or_create_collection(
        name=constants.DEFAULT_COLLECTION_NAME,
        embedding_function=constants.DEFAULT_EF
    )

    # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    # "By default, input text longer than 256 word pieces is truncated."

    docs = [
        rules[:128],
        rules[:256],
        rules[:512],
        rules[:1024],
        rules[:2048],
        rules[:4096],
        rules[:8192],
    ]

    collection.upsert(
        documents=docs,
        ids=[str(x) for x in range(len(docs))]
    )

    res = collection.get(include=['documents', 'embeddings'])

    # all documents 2k or larger appear to have the same embeddings

    for uid, doc, emb in zip(res['ids'], res['documents'], res['embeddings']):
        print(uid, len(doc), np.array(emb[:5]), '...', np.array(emb[-5:]))


if __name__ == '__main__':
    main()
