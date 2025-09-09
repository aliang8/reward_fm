import argparse
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List

import pyrallis
import yaml


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge b into a (mutates a). Scalars and non-dicts in b overwrite."""
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def parse_multiple(config_class):
    # 1) quick-parse only --config_paths
    tmp = argparse.ArgumentParser(add_help=False)
    tmp.add_argument("--config_paths", nargs="*", default=[])
    parsed, remaining_argv = tmp.parse_known_args()

    # 2) load & deep-merge YAMLs in order (later files override earlier ones)
    merged: Dict[str, Any] = {}
    for path in parsed.config_paths:
        with open(path, "r") as f:
            doc = yaml.safe_load(f) or {}
        deep_merge(merged, doc)

    # 3) If we have a merged dict, write it to a temp yaml and let pyrallis load it
    tmp_path = None
    try:
        if merged:
            tf = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
            yaml.safe_dump(merged, tf)
            tf.close()
            tmp_path = tf.name
            cfg = pyrallis.parse(config_class=config_class, config_path=tmp_path, args=remaining_argv)
        else:
            # no yaml files provided — just let pyrallis parse normally from defaults + CLI
            cfg = pyrallis.parse(config_class=config_class, args=remaining_argv)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return cfg