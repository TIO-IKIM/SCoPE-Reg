#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py — evaluate a finished run directory.

    python eval.py runs/camus

Everything comes out of the run directory itself: `config.yaml` says which
model, which dataset variant, which split and which resolution, and
`best_checkpoint.pth.tar` holds the weights. There is no --model flag — the
config is the single source of truth, so a directory can never be evaluated as
something other than what was trained in it.

A cross-validated run (`--fold all`, i.e. save_dir/fold_0, fold_1, ...) is
detected automatically: every fold is evaluated and the table gains a mean±std
row.

Reported per run:

  Registration   mTRE (mm)      mean distance over 5 landmarks (4 frame corners
                                + centre) — the primary metric
                 TransErr (mm)  translation-only L2
                 RotErr (deg)   rotation-only L2 over the Euler triple
  Image          NCC, SSIM      frame vs. the volume resampled at the PREDICTED pose
  Segmentation   Dice, HD95     volume mask resampled at the predicted pose vs.
                                the stored GT slice mask. Dice@GT / HD95@GT are
                                the same comparison at the GROUND-TRUTH pose:
                                the achievable ceiling given resampling, so the
                                predicted numbers are interpretable.
  Cost           Params, GFLOPs, FPS, VRAM (MiB), all at inference, batch 1.

The metric maths all live in metrics/ (one file each) and are the same
functions train.py validates with, so evaluation and training numbers are
directly comparable. The pose pipeline (dof6_to_9d, run_model) is imported from
train.py, so training and evaluation share one pose convention.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import load_config                                   # noqa: E402
from data.pool_dataset import PoolDataset, resolve_data_cfg      # noqa: E402
import metrics                                  # noqa: E402

CKPT_NAMES = ("best_checkpoint.pth.tar", "best_checkpoint.pth", "checkpoint.pth.tar")
CONFIG_NAME = "config.yaml"

# name -> (label, unit, lower_is_better)
REPORT = [
    ("mtre",      "mTRE",     "mm",  True),
    ("trans_err", "TransErr", "mm",  True),
    ("rot_err",   "RotErr",   "deg", True),
    ("ncc",       "NCC",      "",    False),
    ("ssim",      "SSIM",     "",    False),
    ("dice",      "Dice",     "",    False),
    ("hd95",      "HD95",     "mm",  True),
]
# Not tabulated: the same overlap at the GROUND-TRUTH pose. It is the ceiling
# Dice/HD95 could reach, and because the pools store the slice mask that
# preprocessing cut at the GT pose, it also proves sample_slice reproduces the
# preprocessing transform. Anything but 1.0 / 0.0 means they have drifted.
SANITY = [("dice_gt", "Dice@GT", 1.0), ("hd95_gt", "HD95@GT", 0.0)]
COST = [
    ("params_m", "Params", "M"),
    ("gflops",   "GFLOPs", ""),
    ("fps",      "FPS",    ""),
    ("vram_mb",  "VRAM",   "MiB"),
]


# ---------------------------------------------------------------------------
# Run-directory discovery
# ---------------------------------------------------------------------------
def find_checkpoint(run_dir):
    for name in CKPT_NAMES:
        p = run_dir / name
        if p.is_file():
            return p
    return None


def discover_runs(run_dir):
    """-> [(label, dir)]. A directory holding fold_* subdirs expands to them."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no such run directory: {run_dir}")

    if (run_dir / CONFIG_NAME).is_file() and find_checkpoint(run_dir):
        return [(run_dir.name, run_dir)]

    folds = sorted((d for d in run_dir.iterdir()
                    if d.is_dir() and d.name.startswith("fold_")),
                   key=lambda d: int(d.name.split("_")[1]))
    runs = [(d.name, d) for d in folds
            if (d / CONFIG_NAME).is_file() and find_checkpoint(d)]
    if runs:
        return runs

    why = (f"it has {CONFIG_NAME} but no checkpoint ({' / '.join(CKPT_NAMES)})"
           if (run_dir / CONFIG_NAME).is_file()
           else f"it has no {CONFIG_NAME}")
    nested = sorted(str(p.parent.relative_to(run_dir))
                    for p in run_dir.glob("*/**/" + CONFIG_NAME))[:6]
    hint = (f"\nEvaluable run directories below it: "
            + ", ".join(nested) if nested else "")
    raise FileNotFoundError(
        f"{run_dir} is not an evaluable run: {why}, and no fold_*/ "
        f"subdirectory has both.{hint}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _Runner(torch.nn.Module):
    """(frame, vol) -> raw pose output.

    The cost metrics (gflops/fps/vram) call `model(*inputs)`, so wrapping here
    keeps the pose conversion out of the FLOP count.
    """

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.dev = device

    def forward(self, frame, vol):
        out = self.model(frame, vol)
        return out[0] if isinstance(out, (tuple, list)) else out


def load_model(cfg, ckpt_path, volume_shape, device):
    from config import build_model
    model = build_model(cfg, device, volume_shape=volume_shape)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = (ckpt.get("model_state_dict") or ckpt.get("state_dict")
             if isinstance(ckpt, dict) else None)
    if state is None:
        state = ckpt                                   # a bare state_dict
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Never silently evaluate a partially-loaded network: that reads as a
        # bad model rather than a bad checkpoint.
        raise RuntimeError(
            f"checkpoint does not match the model built from {CONFIG_NAME}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys. "
            f"First missing: {list(missing)[:3]}; unexpected: {list(unexpected)[:3]}")
    model.eval()
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    best = (ckpt.get("best_mTRE") or ckpt.get("best_DistErr")
            if isinstance(ckpt, dict) else None)
    return model, epoch, best


# ---------------------------------------------------------------------------
# Cost metrics
# ---------------------------------------------------------------------------
def cost_metrics(runner, vol_shape, hw, device, n_iters):
    d, h, w = vol_shape
    vol = torch.zeros(1, 1, d, h, w, device=device)
    frame = torch.zeros(1, 1, *hw, device=device)
    inputs = (frame, vol)

    out = {"params_m": metrics.count_params(runner.model)["total_millions"]}
    with torch.no_grad():
        out["gflops"] = metrics.gflops(runner, inputs)
        out["vram_mb"] = metrics.max_vram_mb(runner, inputs, device)
        out["fps"] = metrics.fps(runner, inputs, n_iters=n_iters,
                                 warmup=min(10, n_iters), device=device)
    return out


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def evaluate_run(run_dir, args, device):
    from train import dof6_to_9d, run_model
    import train

    cfg_path = run_dir / CONFIG_NAME
    ckpt_path = find_checkpoint(run_dir)
    cfg = load_config(str(cfg_path))

    dc = resolve_data_cfg(None, cfg, default_dataset=cfg["data"]["dataset"])
    ds = PoolDataset(dataset=dc["dataset"], split=args.split, size=dc["size"],
                     pm=dc["pm"], fold=dc["fold"], folds=dc.get("folds"),
                     frame_mode=dc["frame_mode"],
                     data_root=dc["data_root"], return_vol_mask=True)
    if args.limit:
        ds.index = ds.index[:args.limit]

    vol_shape = ds.vol_shape
    hw = ds.slice_shape
    spacing = (cfg["train"]["spacing_mm"] if cfg["train"]["spacing_mm"] is not None
               else ds.spacing_mm)

    # dof6_to_9d builds its FrameRigidTransformer from train.IMG_SIZE;
    # leaving the module default in place would convert the GT with the wrong
    # in-plane size at any resolution other than 128.
    train.IMG_SIZE = tuple(vol_shape)
    train._FRT_FOR_GT = None

    model, epoch, best = load_model(cfg, ckpt_path, vol_shape, device)
    runner = _Runner(model, device)

    who = cfg['model']['variant']
    print(f"\n  {run_dir}")
    print(f"    model    {who}   (epoch {epoch}, best val mTRE "
          f"{best if best is None else round(float(best), 4)})")
    print(f"    data     {ds.describe()}"
          + (f"  [--limit {args.limit}]" if args.limit else ""))
    print(f"    spacing  {spacing:.4f} mm/voxel")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    acc = {k: [] for k, *_ in REPORT + [(s[0], s[1], "", True) for s in SANITY]}
    # Per-sample rows. Statistics that matter (median, p90/p95, success rate,
    # paired tests) live at the SAMPLE level -- 5 folds is far too few for a
    # percentile, ~584 pooled frames is not. Each row also carries its CASE id,
    # because frames from one volume are correlated and a bootstrap has to
    # resample cases, not frames.
    per_sample = []
    _case_re = __import__("re").compile(r"^([A-Za-z]+\d+)")
    def _case_of(i):
        fi, fr = ds.index[i]
        stem = Path(ds.files[fi]).name
        m = _case_re.match(stem)
        return (m.group(1) if m else stem), (fr if fr is not None else -1)
    n = 0
    bar = tqdm(loader, desc=f"    eval {run_dir.name}", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for vol, frame, param, seg, vol_msk in bar:
            vol = vol.to(device).float()
            frame = frame.to(device).float()
            param = param.to(device).float()
            seg = seg.to(device).float()
            vol_msk = vol_msk.to(device).float()
            b = vol.shape[0]

            gt9 = dof6_to_9d(param, device)
            pred9, _ = run_model(model, frame, vol, device)

            slice_pred = metrics.sample_slice(vol, pred9, hw)
            mask_pred = metrics.sample_slice(vol_msk, pred9, hw, mode="nearest")
            mask_gt = metrics.sample_slice(vol_msk, gt9, hw, mode="nearest")

            vals = {
                "mtre":      metrics.mtre(pred9, gt9, spacing=spacing, hw=hw).item(),
                "trans_err": metrics.trans_err(pred9, gt9, spacing=spacing).item(),
                "rot_err":   metrics.rot_err(pred9, gt9).item(),
                "ncc":       metrics.ncc(slice_pred, frame).item(),
                "ssim":      metrics.ssim(slice_pred, frame).item(),
                "dice":      metrics.dice(mask_pred, seg).item(),
                "hd95":      metrics.hd95(mask_pred, seg, spacing=spacing),
                "dice_gt":   metrics.dice(mask_gt, seg).item(),
                "hd95_gt":   metrics.hd95(mask_gt, seg, spacing=spacing),
            }
            for k, v in vals.items():
                acc[k].append((v, b))

            if args.per_sample:
                # reduction='none' where the metric supports it; the image /
                # overlap metrics are batch-reduced, so they are recomputed one
                # sample at a time (584 frames -- the cost is irrelevant).
                d_i = metrics.mtre(pred9, gt9, spacing=spacing, hw=hw, reduction='none')
                x_i = metrics.trans_err(pred9, gt9, spacing=spacing, reduction='none')
                r_i = metrics.rot_err(pred9, gt9, reduction='none')
                for j in range(b):
                    case, fridx = _case_of(n + j)
                    sl = slice(j, j + 1)
                    per_sample.append(dict(
                        idx=n + j, case=case, frame=fridx,
                        mtre=float(d_i[j]),
                        trans_err=float(x_i[j]), rot_err=float(r_i[j]),
                        ncc=float(metrics.ncc(slice_pred[sl], frame[sl])),
                        ssim=float(metrics.ssim(slice_pred[sl], frame[sl])),
                        dice=float(metrics.dice(mask_pred[sl], seg[sl])),
                        hd95=float(metrics.hd95(mask_pred[sl], seg[sl], spacing=spacing)),
                    ))
            n += b
            bar.set_postfix(mtre=f"{_wmean(acc['mtre']):.2f}",
                            dice=f"{_wmean(acc['dice']):.3f}", refresh=False)
    bar.close()

    res = {k: _wmean(v) for k, v in acc.items()}
    res.update(cost_metrics(runner, vol_shape, hw, device, args.fps_iters))
    res["_n"] = n
    res["_model"] = who
    res["_epoch"] = epoch
    res["_spacing_mm"] = spacing
    res["_dataset"] = f"{dc['dataset']}{dc['size'] or ''}_pm{dc['pm']}"
    res["_split"] = args.split
    res["_per_sample"] = per_sample
    return res

def _wmean(pairs):
    """Sample-count-weighted mean; NaN batches (empty masks in HD95) skipped."""
    num = den = 0.0
    for v, b in pairs:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        num += float(v) * b
        den += b
    return float("nan") if den == 0 else num / den

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _agg(fn, results, key):
    """Aggregate one metric across folds, ignoring None/NaN (gflops is None
    when neither fvcore nor thop is installed; HD95 is NaN on empty masks)."""
    vals = [r.get(key) for _, r in results]
    vals = [float(v) for v in vals
            if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return None if not vals else float(fn(vals))

def _fmt(v, digits=4):
    if v is None:
        return "n/a"
    if isinstance(v, float) and np.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def print_table(results):
    """results: [(label, dict)] — one row per fold, plus mean±std when >1."""
    labels = [lab for lab, _ in results]
    wide = max(12, max(len(l) for l in labels) + 2)

    def header(title, cols):
        print(f"\n{title}")
        print("  " + "run".ljust(wide) + "".join(c.rjust(12) for c in cols))
        print("  " + "-" * (wide + 12 * len(cols)))

    acc_cols = [f"{lab}{('[' + u + ']') if u else ''}" for _, lab, u, _ in REPORT]
    header("Accuracy", acc_cols)
    for lab, r in results:
        print("  " + lab.ljust(wide)
              + "".join(_fmt(r[k]).rjust(12) for k, *_ in REPORT))
    if len(results) > 1:
        print("  " + "-" * (wide + 12 * len(acc_cols)))
        cells = []
        for k, *_ in REPORT:
            m, s = _agg(np.mean, results, k), _agg(np.std, results, k)
            cells.append("n/a".rjust(12) if m is None
                         else f"{m:.3f}±{s:.3f}".rjust(12))
        print("  " + "mean±std".ljust(wide) + "".join(cells))

    cost_cols = [f"{lab}{('[' + u + ']') if u else ''}" for _, lab, u in COST]
    header("Cost (inference, batch 1)", cost_cols)
    for lab, r in results:
        print("  " + lab.ljust(wide)
              + "".join(_fmt(r[k], 2).rjust(12) for k, *_ in COST))

    # Sanity: overlap at the GT pose. Constant by construction, so it is a
    # check rather than a result -- flag it loudly when it is not.
    print("\nSanity (overlap at the GROUND-TRUTH pose — expected 1.0 / 0.0):")
    for lab, r in results:
        bits, bad = [], False
        for key, name, expect in SANITY:
            v = r[key]
            off = v is None or np.isnan(v) or abs(v - expect) > 1e-3
            bad |= off
            bits.append(f"{name} {_fmt(v)}")
        print(f"  {lab.ljust(wide)}{'  '.join(bits)}"
              + ("   <-- sample_slice no longer matches preprocessing" if bad else "   ok"))


def write_per_sample(run_dir, results):
    """One row per test frame across every fold -> <run_dir>/per_sample.csv."""
    rows = [dict(r, fold=lab) for lab, res in results
            for r in (res.get("_per_sample") or [])]
    if not rows:
        return None
    cols = ["fold", "idx", "case", "frame", "mtre", "trans_err",
            "rot_err", "dice", "hd95", "ncc", "ssim"]
    path = Path(run_dir) / "per_sample.csv"
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    return path, len(rows)


def write_report(run_dir, results):
    cols = ([k for k, *_ in REPORT] + [k for k, *_ in COST]
            + [k for k, *_ in SANITY])
    out = {"runs": {lab: {k: v for k, v in r.items() if k != "_per_sample"}
                    for lab, r in results}}
    if len(results) > 1:
        out["mean"] = {k: _agg(np.nanmean, results, k) for k in cols}
        out["std"] = {k: _agg(np.nanstd, results, k) for k in cols}

    jpath = Path(run_dir) / "eval_metrics.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)

    cpath = Path(run_dir) / "eval_metrics.csv"
    with open(cpath, "w") as f:
        f.write("run," + ",".join(cols) + "\n")
        for lab, r in results:
            f.write(lab + "," + ",".join(
                "" if r[k] is None else f"{r[k]:.6f}" for k in cols) + "\n")
    return jpath, cpath


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Evaluate a training run directory (config + checkpoint "
                    "are read from it; there is no --model flag).")
    p.add_argument("run_dir", help="e.g. runs/camus, or a directory of fold_* runs")
    p.add_argument("--split", default="test", choices=("test", "val", "train"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="evaluate only the first N samples (smoke test)")
    p.add_argument("--fps-iters", type=int, default=50)
    p.add_argument("--per-sample", default=False, action="store_true",
                   help="also write per_sample.csv (one row per test frame, with "
                        "its case id) -- required for percentiles, success rates "
                        "and paired significance tests")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        runs = discover_runs(args.run_dir)
    except FileNotFoundError as exc:
        sys.exit(f"evaluate.py: {exc}")
    print("=" * 78)
    print(f"Evaluating {args.run_dir}  ({len(runs)} run{'s' if len(runs) > 1 else ''}, "
          f"split={args.split}, device={device})")
    print("=" * 78)

    try:
        results = [(label, evaluate_run(d, args, device)) for label, d in runs]
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"evaluate.py: {exc}")

    print_table(results)
    jpath, cpath = write_report(args.run_dir, results)
    ps = write_per_sample(args.run_dir, results)
    if ps:
        print(f"\nWrote {ps[0]}  ({ps[1]} rows)")
    print(f"\nWrote {jpath}\n      {cpath}")


if __name__ == "__main__":
    main()
