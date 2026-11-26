from __future__ import annotations

from pathlib import Path

import pytest

from sapia.config import (ConfigError, PRESETS, TrainConfig, build_config, get_preset, load_yaml_overrides,
                          parse_train_args)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_yaml_files_mirror_python_presets(name):
    from_yaml = build_config(config_path=str(REPO_ROOT / "configs" / f"{name}.yaml"))
    assert from_yaml.to_dict() == get_preset(name).to_dict()


def test_sapia_yaml_lists_every_option():
    data = load_yaml_overrides(str(REPO_ROOT / "configs" / "sapia.yaml"))
    assert set(data) == set(TrainConfig().to_dict())
    assert set(data["model"]) == set(TrainConfig().to_dict()["model"])


def test_presets():
    big, tiny = get_preset("sapia"), get_preset("tiny")
    assert big.model.num_params() == 1_182_900_224
    assert big.target_tokens_per_step == 1_048_576 and big.compile is True
    assert tiny.model.num_params() < 10_000_000 and tiny.compile is False
    with pytest.raises(ConfigError):
        get_preset("nope")


def test_cli_overrides_and_auto_values():
    cfg = parse_train_args(["--preset", "tiny", "--batch_size", "8", "--model.n_kv_head", "4",
                            "--model.qk_norm", "false", "--max_tokens", "5e6"])
    assert cfg.batch_size == 8 and cfg.model.n_kv_head == 4 and cfg.model.qk_norm is False
    assert cfg.max_tokens == 5_000_000
    assert parse_train_args(["--batch_size", "auto"]).batch_size is None


def test_invalid_head_split_is_rejected():
    with pytest.raises(ConfigError):
        parse_train_args(["--model.n_kv_head", "3"])
