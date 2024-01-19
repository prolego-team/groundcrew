"""
Embedding models and related functionality.
"""


from typing import Any

import torch
import torch.nn.functional as F
import transformers as tfs
from transformers import AutoTokenizer, AutoModel


E5_SMALL_V2 = 'intfloat/e5-small-v2'
E5_BASE_V2 = 'intfloat/e5-base-v2'
E5_LARGE_V2 = 'intfloat/e5-large-v2'


def load_e5(model_name: str, cache_dir_path: str) -> tuple[tfs.BertTokenizerFast, tfs.BertModel]:
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
        text_batch: list[str]
        ) -> torch.Tensor:
    """
    Get embeddings using an e5 tokenizer and model.
    Max token length here is 512.
    """

    # Tokenize the input texts
    batch_dict = tokenizer(text_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**batch_dict)

    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def create_windows(
        batch_dict: dict[str, Any],
        window_tokens: int,
        overlap_tokens: int
        ) -> dict[str, Any]:
    """Expand a batch with overlapping windows."""

    n_batch = batch_dict['input_ids'].shape[0]

    # make windows along the first dimension
    shape = batch_dict['input_ids'].shape
    offset = window_tokens - overlap_tokens

    chunks = []
    for idx in range(0, shape[1], offset):
        chunk = dict(
            input_ids=batch_dict['input_ids'][:, idx:idx + window_tokens],
            token_type_ids=batch_dict['token_type_ids'][:, idx:idx + window_tokens],
            attention_mask=batch_dict['attention_mask'][:, idx:idx + window_tokens]
        )

        if chunk['input_ids'].shape[1] < window_tokens:
            extra = window_tokens - chunk['input_ids'].shape[1]
            # print(shape[1], idx, idx + window_tokens, extra)
            zeros = torch.zeros((n_batch, extra), dtype=torch.int64)
            chunk['input_ids'] = torch.concat([chunk['input_ids'], zeros], dim=1)
            chunk['token_type_ids'] = torch.concat([chunk['token_type_ids'], zeros], dim=1)
            chunk['attention_mask'] = torch.concat([chunk['attention_mask'], zeros], dim=1)

        chunks.append(chunk)

    n_chunks = len(chunks)

    # reassemble by concat along batch dimension
    batch_dict = {
        k: torch.concat([x[k] for x in chunks], dim=0)
        for k in batch_dict
    }
    for v in batch_dict.values():
        assert v.shape[0] == n_batch * n_chunks

    return batch_dict


def pivot_output(outputs: torch.Tensor, n_batch: int) -> torch.Tensor:
    """
    Pivot output from a batch of windows.
    Output can be 2+ dimensions.
    """
    return torch.concat([
        outputs[idx:idx + n_batch, ...]
        for idx in range(0, outputs.shape[0], n_batch)
    ], dim=1)


def e5_embeddings_windowed(
        tokenizer: tfs.BertTokenizerFast,
        model: tfs.BertModel,
        text_batch: list[str],
        window_tokens: int,
        overlap_tokens: int
        ) -> torch.Tensor:
    """
    Get embeddings using an e5 tokenizer and model.
    Calculates windows with a certain overlap amount.
    """

    assert window_tokens <= 512

    n_batch = len(text_batch)

    # Tokenize the input texts
    batch_dict = tokenizer(text_batch, max_length=None, padding=True, truncation=False, return_tensors='pt')

    for k, v in batch_dict.items():
        print(k, v.dtype)

    # shape = batch_dict['input_ids'].shape

    batch_dict = create_windows(batch_dict, window_tokens, overlap_tokens)

    # run model
    with torch.no_grad():
        outputs = model(**batch_dict)

    outputs = outputs.last_hidden_state
    emb_dim = outputs.shape[2]
    attention_mask = batch_dict['attention_mask']

    n_chunks = outputs.shape[0] // n_batch

    # before pivot
    assert outputs.shape == (n_batch * n_chunks, window_tokens, emb_dim)
    assert attention_mask.shape == (n_batch * n_chunks, window_tokens)

    # pivot before average pool
    # outputs:        (n_batch * n_chunks) x window_size x emb_dim -> n_batch x (window_size * n_chunks) x emb_dim
    # attention_mask: (n_batch * n_chunks) x window_size           -> n_batch x (window_size * n_chunks)

    outputs = pivot_output(outputs, n_batch)
    attention_mask = pivot_output(attention_mask, n_batch)

    # after pivot
    assert outputs.shape == (n_batch, window_tokens * n_chunks, emb_dim)
    assert attention_mask.shape == (n_batch, window_tokens * n_chunks)

    # (if we average pool first before pivoting, there will be batch items
    # that go into the average pooling with all-zero masks)

    embeddings = average_pool(outputs, attention_mask)

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def average_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
        ) -> torch.Tensor:
    """average pooling using attention mask"""

    # this is how things were written in the example
    # https://huggingface.co/intfloat/e5-base-v2
    # last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    # return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # This is functionally equivalent and clearer IMO:
    last_hidden = last_hidden_states * attention_mask[:, :, None]
    return torch.sum(last_hidden, dim=1) / torch.sum(attention_mask, dim=1)[:, None]
