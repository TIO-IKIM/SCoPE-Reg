#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One file describes one run: data variant, model and training recipe.

    python train.py --config run.yaml --save_dir out/

The resolved config is written next to the checkpoint, so every checkpoint
records exactly which pool (dataset/size/pm/fold), which architecture and which
loss produced it. Unknown keys are a hard error; defaults live in
DEFAULT_CONFIG and a config only lists what it changes.
"""

import copy
import os
from pathlib import Path

import yaml

# The release fixes the encoder, so there is no encoder registry to consult.
_KNOWN_FUSIONS = ("triplane_coord", "triplane_fssm", "cross_attn")



DEFAULT_CONFIG = {
    "seed": 2032,


    "data": {
        "dataset": "CAMUS",        # CAMUS | ProReg
        "size": None,              # CAMUS: 128|192|384 (None = 128); ProReg: n/a
        "pm": 10,                  # DoF magnitude of the sampled frames (+-pm mm/deg)
        # Total number of folds K. null -> the dataset's own official split.
        "folds": 5,
        # Which fold to run: an int 0..K-1, or "all" to train every fold one
        # after another into save_dir/fold_{i}/ (each keeping its best
        # checkpoint), finishing with a mean +- std summary.
        "fold": 0,
        "frame_mode": "expand",    # expand | random | all
        # Fresh pose per training sample, cut on the fly. TRAIN ONLY: val and
        # test always read the stored frames, so the benchmark is unchanged.
        "resample_train": True,
        "data_root": None,         # None -> $SVR_DATA_ROOT
        # [D,H,W] the model is built for; null -> taken from the pool. When set
        # it is checked against the data and a mismatch is an error.
        "img_size": None,
    },

    "train": {
        "epochs": 500,
        "batch_size": 6,
        "accum_steps": 1,
        "lr": 5.0e-5,
        "weight_decay": 0.01,
        "optimizer": "adamw",      # adam | adamw
        "poly_power": 0.9,         # poly decay: lr * (1 - epoch/epochs)^power
        "val_every": 5,            # epochs between validations
        "num_workers": 10,

        # Flow term on the coordinate field (needs fusion='triplane_coord')
        "flow_loss": False,        # switch
        "lambda_flow": 0.0,        # weight (ignored unless flow_loss)

        # Navier-Cauchy term on the coordinate field
        "physics_loss": False,     # switch
        "lambda_physics": 0.0,     # weight (ignored unless physics_loss)

        # mm/voxel for the reported mm metrics. None -> the pool variant's own
        # spacing (FOV-anchored, follows data.size).
        "spacing_mm": None,
    },

    "model": {
        "d_state": 32,                      # SSM state dim S
        "expand": 2,                        # inner width C' = expand * C
        "reduce_n": None,                   # DepthReduction state dim; null follows d_state
        "reduce_bidirectional": True,       # scan the depth axis both ways
        "fusion": "triplane_coord",         # see _KNOWN_FUSIONS
        "fusion_kwargs": {"only_axial": True},   # only_axial: false = tri-plane
        # Readout. "procrustes" solves for the pose with WKPS; "neck" regresses
        # it with the direct heads, and coord_variant then says what feeds them:
        # 1 = the pooled fused volume, 3 = the coordinate field.
        "variant": "procrustes",
        "coord_variant": 3,
        "pose_from_phi": "procrustes",
        "kabsch_weight": "fusion_conf",     # uniform | fusion_conf | learned
    },
}

# Our model always regresses the 9D (6D-rotation + translation) pose; the

_OPAQUE = {"fusion_kwargs"}


def _deep_merge(base, override, path=""):
    """Merge `override` into `base`; unknown keys raise (typo protection)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        where = f"{path}.{k}" if path else k
        if k not in out:
            raise KeyError(
                f"Unknown config key '{where}'. Allowed here: {sorted(out.keys())}. "
                f"(Schema: config.DEFAULT_CONFIG)")
        if k in _OPAQUE:
            out[k] = copy.deepcopy(v)
        elif isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v, where)
        else:
            out[k] = v
    return out


def derive_out_dof(cfg):
    """Our model's head is always 9D (translation + 6D rotation)."""
    return 9


def load_config(path):
    if not str(path).lower().endswith((".yaml", ".yml")):
        raise ValueError(f"configs are YAML; got '{path}'")
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    # Derived fields (underscore-prefixed) are recomputed, never merged, so a
    # saved snapshot can be loaded back unchanged.
    user = {k: v for k, v in user.items() if not k.startswith("_")}
    cfg = _deep_merge(DEFAULT_CONFIG, user)
    cfg["_out_dof"] = derive_out_dof(cfg)

    t, d = cfg["train"], cfg["data"]

    if d["dataset"] not in ("CAMUS", "ProReg"):
        raise ValueError(f"data.dataset must be CAMUS|ProReg, got '{d['dataset']}'")
    for k in ("lr", "weight_decay", "lambda_flow", "lambda_physics"):
        if isinstance(t[k], str):
            raise ValueError(f"train.{k} is the string {t[k]!r}. Write it as a "
                             f"plain decimal (0.00005) or 5.0e-5 — YAML does "
                             f"not parse 5e-05 as a number.")
    if t["optimizer"] not in ("adam", "adamw"):
        raise ValueError(f"train.optimizer must be adam|adamw, got '{t['optimizer']}'")
    if isinstance(d["size"], str):
        raise ValueError(f"data.size must be an int or null, got the string "
                         f"{d['size']!r}. In YAML write `size: null`.")
    if d["size"] is not None and not isinstance(d["size"], int):
        raise ValueError(f"data.size must be an int or null, got {d['size']!r}")
    if d["folds"] is not None and (not isinstance(d["folds"], int) or d["folds"] < 2):
        raise ValueError(f"data.folds is the number of folds K (int >=2) or null "
                         f"for the official split, got {d['folds']!r}")
    if isinstance(d["fold"], str):
        if d["fold"] != "all":
            raise ValueError(f'data.fold must be an int or "all", got {d["fold"]!r}')
        if d["folds"] is None:
            raise ValueError('data.fold="all" needs data.folds set to K')
    elif not isinstance(d["fold"], int) or d["fold"] < 0:
        raise ValueError(f"data.fold must be a non-negative int or \"all\", "
                         f"got {d['fold']!r}")
    if d["img_size"] is not None:
        if len(d["img_size"]) != 3 or any(int(x) <= 0 for x in d["img_size"]):
            raise ValueError(f"data.img_size must be [D,H,W] of positive ints, "
                             f"got {d['img_size']}")
        d["img_size"] = [int(x) for x in d["img_size"]]


    m = cfg["model"]
    if m["fusion"] not in _KNOWN_FUSIONS:
        raise ValueError(f"model.fusion must be in {_KNOWN_FUSIONS}, "
                         f"got '{m['fusion']}'")
    if m["variant"] not in ("procrustes", "neck"):
        raise ValueError(f"model.variant must be procrustes|neck, "
                         f"got '{m['variant']}'")
    if m["variant"] == "procrustes":
        if m["fusion"] != "triplane_coord":
            raise ValueError("model.variant: procrustes (the WKPS readout) needs "
                             f"model.fusion: triplane_coord, got '{m['fusion']}'")
        if m["kabsch_weight"] not in ("uniform", "fusion_conf", "learned"):
            raise ValueError("model.kabsch_weight must be uniform|fusion_conf|"
                             f"learned, got '{m['kabsch_weight']}'")
    else:
        if m.get("coord_variant") not in (1, 3):
            raise ValueError(
                "model.variant: neck requires model.coord_variant: 1 (pose from "
                "the pooled fused volume) or 3 (pose from the coordinate field); "
                f"got {m.get('coord_variant')!r}")
        if m["coord_variant"] == 3 and m.get("pose_from_phi") != "heads":
            raise ValueError("coord_variant: 3 with variant: neck requires "
                             "pose_from_phi: heads")
    rn = m["reduce_n"]
    if rn is not None and (not isinstance(rn, int) or rn < 1):
        raise ValueError(f"model.reduce_n must be null or a positive int, got {rn!r}")
    # The physics and flow terms penalise the coordinate field; a fusion
    # that emits none has nothing for them to act on.
    if t["physics_loss"] and t["lambda_physics"] > 0 \
            and m["fusion"] != "triplane_coord":
        raise ValueError(
            f"train.physics_loss needs model.fusion='triplane_coord' "
            f"(got '{m['fusion']}'): the term penalises the coordinate field, "
            f"which this fusion does not produce. Set physics_loss: false.")
    if t["flow_loss"] and t["lambda_flow"] > 0 and m["fusion"] != "triplane_coord":
        raise ValueError(f"train.flow_loss needs model.fusion='triplane_coord' "
                         f"(got '{m['fusion']}')")

    return cfg


def save_config(cfg, path):
    """Snapshot the resolved config next to the checkpoint."""
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def config_beside_checkpoint(checkpoint_path, name="config.yaml"):
    return os.path.join(os.path.dirname(checkpoint_path), name)


def run_name(cfg):
    """Short tag for logs: what was trained on what."""
    d = cfg["data"]
    who = cfg["model"]["variant"]
    size = f"{d['size'] or 'native'}"
    return f"{who}_{d['dataset']}{size}_pm{d['pm']}_fold{d['fold']}"


def split_meta(cfg):
    """The fields that pin down the exact data variant a checkpoint saw."""
    d = cfg["data"]
    return {"dataset": d["dataset"], "size": d["size"], "pm": d["pm"],
            "fold": d["fold"], "frame_mode": d["frame_mode"],
            "variant": cfg["model"]["variant"]}


def build_model(cfg, device, volume_shape=None):
    """The ONE place models are constructed."""
    from model import build_model as _build
    return _build(cfg, device, volume_shape=volume_shape)



if __name__ == "__main__":
    import sys
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/cfg.yaml")
    print("run:", run_name(cfg))
    print("out_dof:", cfg["_out_dof"])
    print("data:", cfg["data"])
    print("epochs:", cfg["train"]["epochs"],
          "| bs:", cfg["train"]["batch_size"])
