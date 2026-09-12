import yaml
import pytest
from pathlib import Path

from utils import ROOT, load_params


def test_dvc_yaml_exists_and_valid():
    dvc_yaml_path = ROOT / "dvc.yaml"
    assert dvc_yaml_path.exists()

    with open(dvc_yaml_path, "r", encoding="utf-8") as f:
        dvc_config = yaml.safe_load(f)

    assert "stages" in dvc_config
    stages = dvc_config["stages"]
    assert "data_prep" in stages
    assert "train" in stages

    # Check data_prep stage
    data_prep = stages["data_prep"]
    assert "cmd" in data_prep
    assert "deps" in data_prep
    assert "outs" in data_prep

    # Check train stage
    train = stages["train"]
    assert "cmd" in train
    assert "deps" in train
    assert "outs" in train
    assert "metrics" in train


def test_params_yaml_contains_all_dvc_referenced_params():
    dvc_yaml_path = ROOT / "dvc.yaml"
    with open(dvc_yaml_path, "r", encoding="utf-8") as f:
        dvc_config = yaml.safe_load(f)

    params = load_params()

    for stage_name, stage_info in dvc_config.get("stages", {}).items():
        if "params" in stage_info:
            for param_ref in stage_info["params"]:
                section, key = param_ref.split(".")
                assert section in params, f"Section {section} missing in params.yaml"
                assert key in params[section], f"Key {key} missing in params.yaml[{section}]"
