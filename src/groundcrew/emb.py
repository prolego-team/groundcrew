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


def average_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
        ) -> torch.Tensor:
    """average pooling using attention mask"""

    # this is how things were written in the example
    # last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    # return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # I would write stuff like this I think
    # should be equivalent
    last_hidden = last_hidden_states * attention_mask[..., None]
    return torch.sum(last_hidden, dim=1) / torch.sum(attention_mask, dim=1)[..., None]
