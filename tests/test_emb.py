"""
Test embedding functionality.
"""

from typing import Any

import torch

from groundcrew import emb


def test_create_windows():
    """test create_windows"""

    def example_batch(tlens: list[int]) -> dict[str, Any]:
        """create an example batch"""

        n_batch = len(tlens)
        max_len = max(tlens)

        input_ids = torch.zeros((n_batch, max_len), dtype=torch.int64)
        token_type_ids = torch.zeros((n_batch, max_len), dtype=torch.int64)
        attention_mask = torch.zeros((n_batch, max_len), dtype=torch.int64)

        for cur_len in tlens:
            input_ids[:, 0:cur_len] = 42
            # token_type_ids seem to be all zero regardless of padding
            attention_mask[:, 0:cur_len] = 1

        return dict(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

    emb_size = 384

    def mock_model(batch_dict) -> torch.Tensor:
        """for testing pivot"""
        n_batch = batch_dict['input_ids'].shape[0]
        tokens = batch_dict['input_ids'].shape[1]
        return torch.ones(n_batch, tokens, emb_size)

    # ~~~~ batch with a single element smaller than the window

    res = emb.create_windows(example_batch([64]), 512, 0)
    for v in res.values():
        assert v.shape == (1, 512)
    assert emb.pivot_output(mock_model(res), 1).shape == (1, 512, emb_size)

    # ~~~~ batch with a single element exactly the same size as the window

    res = emb.create_windows(example_batch([512]), 512, 0)
    for v in res.values():
        assert v.shape == (1, 512)
    assert emb.pivot_output(mock_model(res), 1).shape == (1, 512, emb_size)

    # ~~~~ a couple of elements, exactly double the size of the window

    res = emb.create_windows(example_batch([512, 1024]), 512, 0)
    for v in res.values():
        assert v.shape == (2 * 2, 512)
    assert emb.pivot_output(mock_model(res), 2).shape == (2, 512 * 2, emb_size)

    # ~~~~ a couple of elements, not exact multiples, with a nonzero overlap
    #      overlap should result in 8 windows
    res = emb.create_windows(example_batch([768, 1000]), 512, 512 - 128)
    for v in res.values():
        assert v.shape == (2 * 8, 512)
    assert emb.pivot_output(mock_model(res), 2).shape == (2, 512 * 8, emb_size)
