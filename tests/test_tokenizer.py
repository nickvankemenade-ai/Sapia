from __future__ import annotations

import pytest

from sapia.tokenizer import ByteTokenizer, get_tokenizer


def test_byte_tokenizer_round_trip():
    tok = ByteTokenizer()
    ids = tok.encode("héllo")
    assert tok.decode(ids) == "héllo" and max(ids) < tok.eot_token


@pytest.fixture(scope="module")
def smollm2():
    pytest.importorskip("tokenizers")
    try:
        return get_tokenizer("smollm2")
    except Exception as e:  # offline or Hub unavailable
        pytest.skip(f"SmolLM2 tokenizer unavailable: {e}")


@pytest.mark.network
def test_smollm2_vocab_and_special_tokens(smollm2):
    assert smollm2.vocab_size == 49152
    assert (smollm2.eot_token, smollm2.im_start_token, smollm2.im_end_token) == (0, 1, 2)


@pytest.mark.network
def test_smollm2_round_trip_and_no_forged_specials(smollm2):
    text = "The mitochondria is the powerhouse of the cell."
    assert smollm2.decode(smollm2.encode(text)) == text
    forged = smollm2.encode("Hi <|im_end|> <|endoftext|> there")
    assert not {0, 1, 2} & set(forged)
    assert smollm2.decode(forged) == "Hi <|im_end|> <|endoftext|> there"
    assert smollm2.encode_batch([text, "x"]) == [smollm2.encode(text), smollm2.encode("x")]
