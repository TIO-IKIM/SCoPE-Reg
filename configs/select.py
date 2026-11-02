import argparse
import copy
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FIXED_MODEL = {
    "d_state": 32,
    "expand": 2,
    "reduce_n": None,
    "reduce_bidirectional": True,
}
FIXED_TRAIN = {
    "accum_steps": 1,
}

# readout -> the model keys that implement it. Solver arms carry no head
# parameters; head arms carry no solver parameters.
# NOTE: coord_variant must be given explicitly for the head arms. Its default
# is 3 (pose from the coordinate field), so omitting it would silently route a
# head arm into the solver branch instead of failing.
READOUTS = {
    "kabsch":           {"variant": "procrustes"},
    "heads":            {"variant": "neck", "coord_variant": 1},
    "heads_from_coord": {"variant": "neck", "coord_variant": 3,
                         "pose_from_phi": "heads"},
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = (deep_merge(out[k], v)
                  if isinstance(v, dict) and isinstance(out.get(k), dict)
                  else copy.deepcopy(v))
    return out


def build_model_block(arm):
    readout = arm["readout"]
    if readout not in READOUTS:
        sys.exit(f"unknown readout {readout!r}; expected {sorted(READOUTS)}")
    model = dict(FIXED_MODEL)
    model["fusion"] = arm["fusion"]
    if "only_axial" in arm:
        model["fusion_kwargs"] = {"only_axial": arm["only_axial"]}
    else:
        model["fusion_kwargs"] = {}
    model.update(READOUTS[readout])
    if readout == "kabsch":
        model["kabsch_weight"] = arm["kabsch_weight"]
    elif "kabsch_weight" in arm:
        sys.exit(f"kabsch_weight is meaningless for readout {readout!r}")
    return model


def build_train_block(train):
    """Merge the tunable train block with the fixed settings."""
    t = dict(train)
    t.update(FIXED_TRAIN)
    t["flow_loss"] = bool(train.get("flow_loss", False))
    t["physics_loss"] = bool(train.get("physics_loss", False))
    t["lambda_flow"] = float(train.get("lambda_flow", 0.0))
    t["lambda_physics"] = float(train.get("lambda_physics", 0.0))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=("CAMUS", "ProReg"))
    ap.add_argument("--arm", default="wkps", help="ablation arm (default: the reported model)")
    args = ap.parse_args()

    spec = yaml.safe_load((HERE / "final.yaml").read_text())
    if args.arm not in spec["arms"]:
        sys.exit(f"unknown arm {args.arm!r}; expected {sorted(spec['arms'])}")

    merged = deep_merge(spec["common"], spec[args.dataset])
    cfg = {
        "seed": merged["seed"],
        "data": merged["data"],
        "train": build_train_block(merged["train"]),
        "model": build_model_block(spec["arms"][args.arm]),
    }
    yaml.safe_dump(cfg, sys.stdout, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    main()
