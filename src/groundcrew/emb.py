"""
Embedding models and related functionality.
"""

from typing import List, Tuple
import torch

import torch.nn.functional as F

import transformers as tfs
from transformers import AutoTokenizer, AutoModel


E5_SMALL_V2 = 'intfloat/e5-small-v2'
E5_BASE_V2 = 'intfloat/e5-base-v2'
E5_LARGE_V2 = 'intfloat/e5-large-v2'


def load_e5(model_name: str, cache_dir_path: str) -> Tuple[tfs.BertTokenizerFast, tfs.BertModel]:
    """load E5 tokenizer and model"""

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        # device_map='mps',
        cache_dir=cache_dir_path)

    # this ends up being a BertModel
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        # device_map='mps',  # this doesn't work with torch 2.0.0
        cache_dir=cache_dir_path)

    return tokenizer, model


def e5_embeddings(
        tokenizer: tfs.BertTokenizerFast,
        model: tfs.BertModel,
        text_batch: List[str]
        ) -> torch.Tensor:
    """aldskfjhaldkjfh aldjf"""

    # Tokenize the input texts
    batch_dict = tokenizer(text_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**batch_dict)

    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def e5_embeddings_windowed(
        tokenizer: tfs.BertTokenizerFast,
        model: tfs.BertModel,
        text_batch: List[str],
        window_tokens: int,
        overlap_tokens: int
    ) -> torch.Tensor:
    """adkfj haldkjf halsdkjfh """

    n_batch = len(text_batch)

    # Tokenize the input texts
    batch_dict = tokenizer(text_batch, max_length=None, padding=True, truncation=False, return_tensors='pt')

    # Make windows along dimension 1
    # and pivot into a new batch
    shape = batch_dict['input_ids'].shape

    offset = window_tokens - overlap_tokens
    print(shape)
    chunks = []
    for idx in range(0, shape[1], offset):
        chunk = dict(
            input_ids=batch_dict['input_ids'][:, idx:idx + window_tokens],
            token_type_ids=batch_dict['token_type_ids'][:, idx:idx + window_tokens],
            attention_mask=batch_dict['attention_mask'][:, idx:idx + window_tokens]
        )

        if chunk['input_ids'].shape[1] < window_tokens:
            continue
            extra = window_tokens - chunk['input_ids'].shape[1]
            print(shape[1], idx, idx + window_tokens, extra)
            zeros = torch.zeros((n_batch, extra), dtype=torch.int64)
            chunk['input_ids'] = torch.concat([chunk['input_ids'], zeros], axis=1)
            chunk['token_type_ids'] = torch.concat([chunk['token_type_ids'], zeros], axis=1)
            chunk['attention_mask'] = torch.concat([chunk['attention_mask'], zeros], axis=1)

        chunks.append(chunk)

    # concat along batch dimension
    n_chunks = len(chunks)

    batch_dict = {
        k: torch.concat([x[k] for x in chunks], axis=0)
        for k in batch_dict
    }

    for v in batch_dict.values():
        print(v.shape)
        assert v.shape[0] == n_batch * n_chunks

    with torch.no_grad():
        outputs = model(**batch_dict)

    # TODO: how to handle all masked stuff here??? need to think

    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # pivot and aggregate

    embeddings = torch.mean(
        torch.concat(
            [
                embeddings[idx:idx + n_batch, :][:, :, None]
                for idx in range(0, embeddings.shape[0], n_batch)
            ], axis=2
        ),
        axis=2
    )

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def average_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
        ) -> torch.Tensor:
    """average pooling using attention mask"""

    # this is how things were written in the example
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # I would write stuff like this I think
    # should be equivalent
    # last_hidden = last_hidden_states * attention_mask[..., None]
    # return torch.sum(last_hidden, dim=1) / torch.sum(attention_mask, dim=1)[..., None]
