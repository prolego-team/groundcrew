"""
Figure out how chromadb embedding stuff works.
"""

import click
import chromadb
import numpy as np
import transformers as tfs

from groundcrew import code
from groundcrew import emb as ef


@click.command()
@click.option(
    '--models_dir_path', '-m',
    required=True,
    default='.models_cache',
    help='Directory to download / load the embedding model from.'
)
def main(models_dir_path: str):
    """main program"""

    # ~~~~ prepare a set of documents

    #  can replace with another text document
    with open('mtgrules.txt', 'r') as f:
        rules = f.readlines()
    rules = ''.join(rules)

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

    client = chromadb.PersistentClient('baloney')

    # ~~~~ collection with default embedding function

    collection = client.get_or_create_collection(
        name=code.DEFAULT_COLLECTION_NAME,
        embedding_function=code.DEFAULT_EF
    )
    collection.upsert(
        documents=docs,
        ids=[str(x) for x in range(len(docs))]
    )

    res = collection.get(include=['documents', 'embeddings'])
    ids, docs, embs = res['ids'], res['documents'], res['embeddings']
    print('default embedding function')
    for uid, doc, emb in zip(ids, docs, embs):
        print(uid, len(doc), np.array(emb[:5]), '...', np.array(emb[-5:]))

    # all embeddings made from docs bigger than 1k are identical
    for idx_a, emb_a in enumerate(embs):
        for idx_b, emb_b in enumerate(embs):
            if idx_a != idx_b and (idx_a < 4 or idx_b < 4):
                assert emb_a != emb_b, (idx_a, idx_b)
            else:
                assert emb_a == emb_b, (idx_a, idx_b)

    # ~~~~ collection with custom embedding function

    tokenizer, model = ef.load_e5(
        model_name=ef.E5_SMALL_V2,
        cache_dir_path=models_dir_path
    )

    emb_func = E5EmbeddingFunction(tokenizer, model)

    collection = client.get_or_create_collection(
        name=code.DEFAULT_COLLECTION_NAME,
        embedding_function=emb_func
    )
    collection.upsert(
        documents=docs,
        ids=[str(x) for x in range(len(docs))]
    )

    res = collection.get(include=['documents', 'embeddings'])
    ids, docs, embs = res['ids'], res['documents'], res['embeddings']
    print('custom embedding function')
    for uid, doc, emb in zip(ids, docs, embs):
        print(uid, len(doc), np.array(emb[:5]), '...', np.array(emb[-5:]))

    # all embeddings are distinct
    for idx_a, emb_a in enumerate(embs):
        for idx_b, emb_b in enumerate(embs):
            if idx_a != idx_b:
                assert emb_a != emb_b, (idx_a, idx_b)
            else:
                assert emb_a == emb_b, (idx_a, idx_b)


class E5EmbeddingFunction(chromadb.EmbeddingFunction):
    """Embedding function for chromadb using E5."""

    def __init__(self, tokenizer: tfs.BertTokenizerFast, model: tfs.BertModel):
        """Constructor"""
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        """Create embeddings using E5."""

        emb_tensor = ef.e5_embeddings_windowed(
            tokenizer=self.tokenizer,
            model=self.model,
            text_batch=input,
            window_tokens=512,
            overlap_tokens=256
        )

        return [x.tolist() for x in emb_tensor]


if __name__ == '__main__':
    main()
