"""The mix pipeline end to end on small local files (offline): prepare_mix, then MixtureLoader."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sapia.data import DataError, MixtureLoader, mixture_val_sets, read_shard
from sapia.prepare_mix import MixConfig, read_mix, run_mix
from sapia.sources import SourceSpec
from sapia.tokenizer import ByteTokenizer

from conftest import make_docs, write_parquet

TOK = ByteTokenizer()


def _quiet(_msg: str) -> None:
    pass


@pytest.fixture
def local_mix(tmp_path: Path):
    """Two text sources and a chat source as local files; returns (phases, out_dir)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for i in range(2):
        write_parquet(a / f"a{i}.parquet", [d.replace("doc", "alpha") for d in make_docs(150, seed=i)])
        write_parquet(b / f"b{i}.parquet", [d.replace("doc", "beta") for d in make_docs(150, seed=10 + i)])
    chat = tmp_path / "chat.jsonl"
    with open(chat, "w", encoding="utf-8") as fh:
        for i in range(60):
            fh.write(json.dumps({"messages": [{"role": "user", "content": f"question {i}?"},
                                              {"role": "assistant", "content": f"answer {i}."}]}) + "\n")
    phases = {
        "main": [SourceSpec("alpha", 0.75, "", str(a / "*.parquet")),
                 SourceSpec("beta", 0.25, "", str(b / "*.parquet"))],
        "anneal": [SourceSpec("beta_late", 0.5, "", str(b / "*.parquet"), file_mod=2, file_keep=(1,)),
                   SourceSpec("chat", 0.5, "", str(chat), "chat")],
    }
    return phases, tmp_path / "mix"


def _cfg(out_dir: Path, **kw) -> MixConfig:
    base = dict(out_dir=str(out_dir), total_tokens=40_000, anneal_fraction=0.25, tokenizer="bytes",
                val_every=10, num_workers=1, shard_tokens=3000)
    base.update(kw)
    return MixConfig(**base)


def test_prepare_mix_writes_each_source_with_its_target(local_mix):
    phases, out = local_mix
    mix = run_mix(_cfg(out), phases, log=_quiet)
    assert read_mix(out) == mix
    main = {s["name"]: s for s in mix["phases"]["main"]["sources"]}
    assert mix["phases"]["main"]["tokens"] == 30_000 and mix["phases"]["anneal"]["tokens"] == 10_000
    assert main["alpha"]["target_tokens"] == 22_500 and main["beta"]["target_tokens"] == 7_500
    for s in main.values():
        assert s["prepared"] and s["target_tokens"] <= s["train_tokens"] < s["target_tokens"] + 200
    assert (out / "main" / "alpha" / "manifest.json").exists()


def test_chat_documents_contain_real_special_tokens(local_mix):
    phases, out = local_mix
    run_mix(_cfg(out, phases="anneal"), phases, log=_quiet)
    shard = sorted((out / "anneal" / "chat").glob("train_*.bin"))[0]
    toks = read_shard(shard).tolist()
    first = toks[: toks.index(TOK.eot_token) + 1]
    expected = ([TOK.im_start_token] + TOK.encode("user\nquestion 1?") + [TOK.im_end_token] + TOK.encode("\n")
                + [TOK.im_start_token] + TOK.encode("assistant\nanswer 1.") + [TOK.im_end_token]
                + TOK.encode("\n") + [TOK.eot_token])
    assert first == expected                     # doc 0 went to the val split (val_every=10)


def test_only_and_resume_leave_other_sources_alone(local_mix):
    phases, out = local_mix
    run_mix(_cfg(out, only="beta"), phases, log=_quiet)
    assert not (out / "main" / "alpha").exists()
    before = sorted(p.name for p in (out / "main" / "beta").glob("*.bin"))
    mix = run_mix(_cfg(out), phases, log=_quiet)            # rerun everything: beta is already done
    assert sorted(p.name for p in (out / "main" / "beta").glob("*.bin")) == before
    assert all(s["prepared"] for s in mix["phases"]["main"]["sources"])


def _prepared(local_mix):
    phases, out = local_mix
    run_mix(_cfg(out), phases, log=_quiet)
    return out


def test_mixture_draws_follow_the_shares(local_mix):
    out = _prepared(local_mix)
    loader = MixtureLoader(out, "main", block_size=16, batch_size=64, seed=3)
    counts = np.zeros(2)
    for _ in range(60):
        tokens, draws = loader.next_batch()
        assert tokens.shape == (64, 17)
        counts += np.bincount(draws, minlength=2)
    frac = counts / counts.sum()
    assert loader.names == ["alpha", "beta"]
    assert abs(frac[0] - 0.75) < 0.03


def test_mixture_rows_come_from_the_drawn_source(local_mix):
    out = _prepared(local_mix)
    loader = MixtureLoader(out, "main", block_size=64, batch_size=32, seed=1)
    tokens, draws = loader.next_batch()
    for row, src in zip(tokens, draws):
        text = bytes(int(t) for t in row if t < 256).decode("utf-8", "replace")
        word = "alpha" if loader.names[src] == "alpha" else "beta"
        other = "beta" if word == "alpha" else "alpha"
        assert other not in text


def test_mixture_resumes_exactly(local_mix):
    out = _prepared(local_mix)
    ref = MixtureLoader(out, "main", block_size=16, batch_size=8, seed=5)
    batches = [ref.next_batch()[0] for _ in range(30)]
    first = MixtureLoader(out, "main", block_size=16, batch_size=8, seed=5)
    for _ in range(12):
        first.next_batch()
    state = json.loads(json.dumps(first.state()))           # survives a checkpoint round trip
    resumed = MixtureLoader(out, "main", block_size=16, batch_size=8, seed=5, state=state)
    for i in range(12, 30):
        assert np.array_equal(resumed.next_batch()[0], batches[i])


def test_mixture_wraps_a_source_into_its_next_epoch(local_mix):
    out = _prepared(local_mix)
    loader = MixtureLoader(out, "main", block_size=16, batch_size=64, seed=0)
    n_beta = loader.shards[1].n_windows
    for _ in range(4 * n_beta // 16 + 5):
        loader.next_batch()
    assert loader.state()["positions"]["beta"][0] >= 1       # beta was used up at least once


def test_mixture_state_for_another_phase_is_rejected(local_mix):
    out = _prepared(local_mix)
    state = MixtureLoader(out, "main", 16, 4, 0).state()
    with pytest.raises(DataError):
        MixtureLoader(out, "anneal", 16, 4, 0, state=state)


def test_validation_sets_per_source(local_mix):
    out = _prepared(local_mix)
    vals = mixture_val_sets(out, "main", block_size=16)
    assert set(vals) == {"alpha", "beta"}
    assert vals["alpha"][0] == 0.75 and vals["alpha"][1].n_windows > 0
