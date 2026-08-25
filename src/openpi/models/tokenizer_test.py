import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_batch_tokenize_matches_scalar_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=24)
    prompts = ["pick_up block", "open drawer"]
    states = [np.asarray([0.1, -0.2]), np.asarray([0.5, 0.9])]

    batch_tokens, batch_masks = tokenizer.tokenize_batch(prompts, states, num_threads=2)
    scalar = [tokenizer.tokenize(prompt, state) for prompt, state in zip(prompts, states, strict=True)]

    np.testing.assert_array_equal(batch_tokens, np.stack([item[0] for item in scalar]))
    np.testing.assert_array_equal(batch_masks, np.stack([item[1] for item in scalar]))


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
