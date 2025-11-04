#!/usr/bin/env python3
"""
Generate a newline-separated command list for sweeps.

Usage:
  uv run python3 scripts/generate_sweep.py --sweep_config sweeps/rewind_transformer.yaml --out sweep_commands.txt
  uv run python3 scripts/generate_sweep.py --sweep_config sweeps/qwen.yaml --out sweep_commands.txt

"""

import argparse
import itertools
import json
import os
import shlex
import sys
from typing import Any, Dict, List

try:
    import yaml
except Exception:
    print("Please install pyyaml: uv pip install pyyaml", file=sys.stderr)
    raise

KEY_SHORTEN_MAP = {
    "data.sample_type_ratio": "st",
    "model.train_progress_head": "prog",
    "model.train_preference_head": "pref",
    "model.train_similarity_head": "sim",
    "model.train_success_head": "succ",
    "data.pairwise_progress": "2frames",
}


def _sanitize_val(v: Any) -> str:
    if isinstance(v, list):
        return "_".join(_sanitize_val(x) for x in v)
    if isinstance(v, bool):
        return "t" if v else "f"
    s = str(v)
    # Replace non-alnum with dash
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in [".", ":", "-"]:
            out.append("-")
        elif ch in [" ", ",", "/", "[", "]", "{", "}"]:
            out.append("-")
        else:
            out.append("-")
    # Collapse multiple dashes
    res = "".join(out)
    while "--" in res:
        res = res.replace("--", "-")
    return res.strip("-")


def _make_exp_name(model_tag: str, keys: List[str], values: List[Any]) -> str:
    parts: List[str] = [model_tag]
    for k, v in zip(keys, values):
        if k in KEY_SHORTEN_MAP:
            key_short = KEY_SHORTEN_MAP[k]
            parts.append(f"{key_short}-{_sanitize_val(v)}")
    return "_".join(parts)


def _format_override_arg(key: str, value: Any) -> str:
    """Format a single override as a CLI argument string.

    - Uses dot-path keys like --data.sample_type_ratio
    - Serializes lists/dicts to JSON to preserve structure
    - Lowercases booleans to match YAML/CLI expectations
    """
    if isinstance(value, (list, dict)):
        val_str = json.dumps(value)
    elif isinstance(value, bool):
        val_str = "true" if value else "false"
    else:
        val_str = str(value)

    # Quote if contains whitespace or special chars
    # Keep brackets unquoted for some CLIs, but quoting JSON is safer overall
    # Here we quote everything except simple alnum/._-/:
    if any(ch.isspace() for ch in val_str) or any(ch in val_str for ch in ["[", "]", "{", "}", ",", '"', "'"]):
        val_str = shlex.quote(val_str)

    return f"--{key} {val_str}"


def _build_command(use_accelerate: bool, config_paths: List[str], overrides: Dict[str, Any]) -> str:
    # Hard-coded base commands
    if use_accelerate:
        # Hard-coded accelerate flags (see scripts/train.sh)
        accel = [
            "uv",
            "run",
            "accelerate",
            "launch",
            "--config_file",
            "rfm/configs/fsdp.yaml",
            "--num_processes=2",
            "train.py",
        ]
        base_cmd = " ".join(accel)
    else:
        base_cmd = "uv run python3 train.py"

    lines: List[str] = [base_cmd + " \\"]

    override_items = list(overrides.items())

    if config_paths:
        config_paths_str = " ".join(config_paths)
        if override_items:
            lines.append(f"    --config_paths {config_paths_str} \\")
        else:
            lines.append(f"    --config_paths {config_paths_str}")
    for i, (k, v) in enumerate(override_items):
        arg = _format_override_arg(k, v)
        if i < len(override_items) - 1:
            lines.append(f"    {arg} \\")
        else:
            lines.append(f"    {arg}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_config", type=str, required=True, help="Path to sweep YAML config")
    parser.add_argument("--out", type=str, required=True, help="Output text file for commands")
    args = parser.parse_args()

    with open(args.sweep_config, "r") as f:
        cfg = yaml.safe_load(f)

    use_accelerate: bool = bool(cfg.get("use_accelerate", False))
    config_paths: List[str] = cfg.get("config_paths", [])
    sweep_overrides: Dict[str, List[Any]] = cfg.get("sweep_overrides", {}) or {}
    combine_mode: str = str(cfg.get("combine", "cartesian")).lower().strip()
    groups: List[Dict[str, Any]] = cfg.get("groups", []) or []

    # Helper: build list of dict combos for a given sweep_overrides and combine mode
    def _build_group_combos(
        group_sweep: Dict[str, List[Any]], mode: str, zip_keys: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        items = list(group_sweep.items())
        if not items:
            return [{}]
        # Determine key order: explicit zip_keys if provided; else insertion order
        if zip_keys:
            keys_local = zip_keys
            values_lists_local = [group_sweep[k] for k in keys_local]
        else:
            keys_local = [k for k, _ in items]
            values_lists_local = [v for _, v in items]
        if mode == "zip":
            lengths = [len(v) for v in values_lists_local]
            if len(set(lengths)) != 1:
                raise ValueError(f"combine=zip requires equal-length lists, got lengths={lengths}")
            combos_iter = zip(*values_lists_local)
        elif mode == "cartesian":
            combos_iter = itertools.product(*values_lists_local)
        return [dict(zip(keys_local, combo)) for combo in combos_iter]

    # Build per-group combos
    group_combos_lists: List[List[Dict[str, Any]]] = []
    # Include top-level sweep_overrides as an implicit group
    if sweep_overrides:
        group_combos_lists.append(_build_group_combos(sweep_overrides, combine_mode))
    # Include explicit groups
    for g in groups:
        g_sw = g.get("sweep_overrides", {}) or {}
        g_mode = str(g.get("combine", "cartesian"))
        g_zip_keys = g.get("zip_keys", None)
        group_combos_lists.append(_build_group_combos(g_sw, g_mode, g_zip_keys))

    # If no groups defined, create an empty group to produce a single run
    if not group_combos_lists:
        group_combos_lists = [[{}]]

    # Cartesian product across groups, merging dicts
    merged_combos: List[Dict[str, Any]] = []
    for combo_tuple in itertools.product(*group_combos_lists):
        merged: Dict[str, Any] = {}
        for d in combo_tuple:
            # Later groups can override earlier keys if duplicated
            merged.update(d)
        merged_combos.append(merged)

    # Determine overall keys ordering for exp_name (stable: sort by key)
    all_keys_sorted = sorted(set(k for d in merged_combos for k in d.keys()))

    # Determine model tag from config paths
    model_tag = "rewind" if any("rewind_transformer_config.yaml" in p for p in config_paths) else "rfm"

    commands: List[str] = []
    for combo_overrides in merged_combos:
        # Build exp name using sorted keys for stability
        values_for_name = [combo_overrides[k] for k in all_keys_sorted]
        exp_name = _make_exp_name(model_tag, all_keys_sorted, values_for_name) if all_keys_sorted else model_tag
        combo_overrides_with_name = {**combo_overrides, "training.exp_name": exp_name}
        cmd = _build_command(use_accelerate, config_paths, combo_overrides_with_name)
        commands.append(cmd)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w") as f:
        f.write("\n".join(commands) + "\n")

    print(f"Wrote {len(commands)} commands to {args.out}")


if __name__ == "__main__":
    main()
