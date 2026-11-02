#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pool_dataset.py — THE dataset layer for the pool-format CAMUS/ProReg data.

The model
reads through this module, so a change to the data contract happens in exactly
one place.

POOL FORMAT
-----------
One pkl per (case, view) bundling the volume ONCE plus its N sampled frames:

    (vol[Z,Y,X], msk[Z,Y,X], slices[N,H,W], slice_masks[N,H,W], dofs[N,6])

The frames were sampled at preprocessing time with a fixed seed, so the poses
are identical across runs and across models -- the benchmark is deterministic.
Volumes are already percentile-normalised to [0,1] and the frames were cut from
those normalised volumes, so NO further normalisation happens here (re-running
a min-max normalise on the frame would silently rescale it against a different
reference than the volume it must be registered to).

LAYOUT ON DISK
--------------
    <root>/CAMUS{size}/pool_pm{pm}/patient0001_2CH_32x128x128.pkl
    <root>/CAMUS{size}/manifests/official_{train,val,test}.txt
    <root>/ProReg/pool_pm{pm}/case000000_subA_40x80x96.pkl
    <root>/ProReg/manifests/fold{k}_{train,val,test}.txt

`size` selects the CAMUS resolution (128/192/384, FOV-anchored: the physical
field of view is constant, only the sampling density changes). `pm` selects the
DoF magnitude the frames were sampled with (+-pm mm/deg, uniform). ProReg has a
single native resolution (40x80x96 @ 0.8mm), so `size` does not apply there.

SPLITS
------
Manifests hold CASE IDs, never file names -- one line per case. Splits are
therefore decoupled from the pool, and regrouping (k-fold) never requires
regenerating data. Grouping by case also guarantees that the two views of one
CAMUS patient (2CH/4CH) resp. the two sub-volumes of one ProReg case
(subA/subB) never straddle a split boundary, which would leak the same
anatomy into train and test.

SAMPLE EXPANSION
----------------
frame_mode='expand' (default) turns each (file, frame_index) pair into its own
sample, so a pool of 1000 CAMUS files with 4 frames each yields 4000 samples.
That matches the sample count of the older one-file-per-frame layout, so epoch
sizes stay comparable to earlier runs, and evaluation is fully deterministic.

CONTRACT
--------
PoolDataset yields, positionally (every consumer indexes data[0..3]):

    data[0]  vol    (1, Z, Y, X)   float32, [0,1]
    data[1]  frame  (1, H, W)      float32, [0,1]
    data[2]  dof    (6,)           float32, [tx,ty,tz,rx,ry,rz] mm/deg
    data[3]  seg    (1, H, W)      float32, frame segmentation label

"""

import argparse
import os
import os.path as osp
import pickle
import re
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

# Override with the SVR_DATA_ROOT environment variable, or per-run via
# `data.data_root` in the config.
DEFAULT_DATA_ROOT = os.environ.get("SVR_DATA_ROOT", "./datasets")

# CAMUS is FOV-anchored on 128px @ 0.616mm; every other resolution samples the
# SAME physical window more finely -> spacing = FOV / size.
CAMUS_FOV_MM = 128 * 0.616          # 78.848 mm

DATASET_SPECS = {
    "CAMUS": {
        "dir": "CAMUS{size}",
        "sizes": (128, 192, 384),
        "default_size": 128,
        # CAMUS ships an official split (folds: null); k-fold manifests are
        # generated alongside it by data/make_manifests.py.
        "has_official": True,
        "case_re": re.compile(r"^(patient\d+)_"),
        "spacing_mm": lambda size: CAMUS_FOV_MM / size,
    },
    "ProReg": {
        "dir": "ProReg",
        "sizes": (None,),                      # single native resolution
        "default_size": None,
        "has_official": False,                 # no official split exists
        "case_re": re.compile(r"^(case\d+)_"),
        "spacing_mm": lambda size: 0.8,        # native, unresampled
    },
}

# Manifest layout.
#   folds=None -> manifests/official_{split}.txt          (dataset's own split)
#   folds=K    -> manifests/k{K}/fold{i}_{split}.txt      (i-th of K folds)
# K is part of the path: a 5-fold and a 10-fold set describe different splits
# and must not share filenames.
OFFICIAL_PATTERN = "official_{split}.txt"
FOLD_DIR = "k{folds}"
FOLD_PATTERN = "fold{fold}_{split}.txt"

KNOWN_DATASETS = tuple(DATASET_SPECS)
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _spec(dataset):
    if dataset not in DATASET_SPECS:
        raise ValueError(f"dataset must be one of {KNOWN_DATASETS}, got '{dataset}'")
    return DATASET_SPECS[dataset]


def _check_size(dataset, size):
    """Normalise + validate `size`; returns the effective size for `dataset`."""
    spec = _spec(dataset)
    if size is None:
        return spec["default_size"]
    if spec["default_size"] is None:
        # ProReg: a size is meaningless here. Accept it silently only if it is
        # the native in-plane extent, otherwise say so instead of loading data
        # that does not match what the caller thinks it asked for.
        raise ValueError(
            f"dataset '{dataset}' has a single native resolution and takes no "
            f"--size (got {size}). Drop the argument.")
    if size not in spec["sizes"]:
        raise ValueError(f"size must be one of {spec['sizes']} for '{dataset}', got {size}")
    return size


def dataset_dir(dataset, size=None, data_root=None):
    size = _check_size(dataset, size)
    root = data_root or DEFAULT_DATA_ROOT
    return osp.join(root, _spec(dataset)["dir"].format(size=size))


def pool_dir(dataset, size=None, pm=10, data_root=None):
    return osp.join(dataset_dir(dataset, size, data_root), f"pool_pm{int(pm)}")


def manifest_path(dataset, split, size=None, fold=0, folds=None, data_root=None):
    """Path of the manifest for `split`.

    folds=None -> the dataset's official split (error if it has none).
    folds=K    -> manifests/k{K}/fold{fold}_{split}.txt
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    mdir = osp.join(dataset_dir(dataset, size, data_root), "manifests")
    if folds is None:
        if not _spec(dataset)["has_official"]:
            raise ValueError(
                f"'{dataset}' has no official split; set data.folds to the "
                f"number of folds (e.g. 5) and data.fold to which one to run.")
        return osp.join(mdir, OFFICIAL_PATTERN.format(split=split))
    return osp.join(mdir, FOLD_DIR.format(folds=int(folds)),
                    FOLD_PATTERN.format(fold=int(fold), split=split))


def spacing_mm(dataset, size=None):
    """Physical voxel spacing of the pool data — metrics in mm need this."""
    return _spec(dataset)["spacing_mm"](_check_size(dataset, size))


def read_manifest(path):
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"missing split manifest: {path}\n"
            f"(CAMUS manifests are written by camus_1.py; ProReg "
            f"manifests by `python -m data.make_manifests --dataset ProReg`)")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def available_k(dataset, size=None, data_root=None):
    """K values (fold counts) that have a manifest directory, sorted."""
    mdir = osp.join(dataset_dir(dataset, size, data_root), "manifests")
    if not osp.isdir(mdir):
        return []
    ks = []
    for fn in os.listdir(mdir):
        m = re.match(r"k(\d+)$", fn)
        if m and osp.isdir(osp.join(mdir, fn)):
            ks.append(int(m.group(1)))
    return sorted(ks)


def folds_present(dataset, folds, size=None, data_root=None):
    """Fold indices of a K-fold set that have a complete train/val/test trio."""
    mdir = osp.join(dataset_dir(dataset, size, data_root), "manifests",
                    FOLD_DIR.format(folds=int(folds)))
    if not osp.isdir(mdir):
        return []
    names = set(os.listdir(mdir))
    return [i for i in range(int(folds))
            if all(FOLD_PATTERN.format(fold=i, split=sp) in names for sp in SPLITS)]


def resolve_folds(dataset, folds, fold=0, size=None, data_root=None):
    """The concrete fold indices this run trains.

    folds=None           -> [None]   the dataset's official split
    folds=K, fold="all"  -> [0..K-1] train every fold, one after another
    folds=K, fold=i      -> [i]      just that one
    """
    if folds is None:
        return [None]

    have = folds_present(dataset, folds, size, data_root)
    if not have:
        ks = available_k(dataset, size, data_root)
        raise ValueError(
            f"no {folds}-fold manifests for '{dataset}'"
            + (f" (available: {ks}-fold)" if ks else "")
            + ". Generate them: python -m data.make_manifests --dataset "
            + f"{dataset}" + (f" --size {size}" if size else "")
            + f" --folds {folds}")

    if isinstance(fold, str):
        if fold != "all":
            raise ValueError(f'data.fold must be an int or "all", got {fold!r}')
        missing = [i for i in range(int(folds)) if i not in have]
        if missing:
            raise ValueError(f"fold=all needs all {folds} folds, missing {missing}")
        return list(range(int(folds)))

    fold = int(fold)
    if fold not in have:
        raise ValueError(f"fold={fold} not available for {folds}-fold "
                         f"'{dataset}' (have {have})")
    return [fold]


def manifests_for(split):
    """Which manifest backs `split`. Splits are always three-way (7:1:2), so
    train, val and test are disjoint and the test set is never touched during
    model selection."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return [split]


def pool_files(dataset, split, size=None, pm=10, fold=0, folds=None,
               data_root=None):
    """All pool files whose case ID is listed in the split's manifest(s)."""
    pdir = pool_dir(dataset, size, pm, data_root)
    if not osp.isdir(pdir):
        script = "camus_1.py" if dataset == "CAMUS" else "proreg_1.py"
        raise FileNotFoundError(
            f"missing pool directory: {pdir}\n"
            f"(build it with data/preprocess/{script} --size/--pm)")
    cases = set()
    for part in manifests_for(split):
        cases |= set(read_manifest(
            manifest_path(dataset, part, size, fold, folds, data_root)))
    case_re = _spec(dataset)["case_re"]

    files, seen = [], set()
    for fn in sorted(os.listdir(pdir)):
        if not fn.endswith(".pkl"):
            continue
        m = case_re.match(fn)
        if m and m.group(1) in cases:
            files.append(osp.join(pdir, fn))
            seen.add(m.group(1))

    if not files:
        raise FileNotFoundError(
            f"no pool files in {pdir} matched the {len(cases)} cases of split "
            f"'{split}' — is the manifest for a different dataset?")
    missing = cases - seen
    if missing:
        print(f"[pool_dataset] WARNING: {len(missing)} case(s) in the '{split}' "
              f"manifest have no pool file (e.g. {sorted(missing)[:3]})")
    return files


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_pool_pkl(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if len(payload) != 5:
        raise ValueError(
            f"{path}: expected a 5-tuple pool file "
            f"(vol, msk, slices, slice_masks, dofs), got {len(payload)} entries. "
            f"This looks like the OLD one-frame-per-file format — rebuild the "
            f"pool with the current *_preprocess_1.py.")
    return payload


class _LRU(OrderedDict):
    """Tiny bounded cache: with frame_mode='expand' the same file is read once
    per frame, so even a small cache removes most of the repeated I/O."""

    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize

    def get_or_load(self, path, loader):
        if self.maxsize <= 0:
            return loader(path)
        if path in self:
            self.move_to_end(path)
            return self[path]
        val = loader(path)
        self[path] = val
        if len(self) > self.maxsize:
            self.popitem(last=False)
        return val


# ---------------------------------------------------------------------------
# On-the-fly frame resampling (training only)
#
# The pools store NUM_SLICES_PER_VIEW poses per volume, drawn once at
# preprocessing time with a fixed seed. That makes the training set a finite
# set of memorisable (volume, pose) pairs. Since the task is self-supervised,
# a FRESH pose can be drawn and its slice cut every time a training sample is
# requested — the training distribution becomes continuous and the stored
# frames stay untouched for val/test.
#
# `_SliceCutter` is a line-for-line port of the FrameRigidTransformer in
# data/preprocess/camus_1.py (same grid, same dof2mat, align_corners=True,
# padding_mode='zeros'), so an on-the-fly frame is drawn from exactly the
# distribution the stored frames came from. Verified by re-cutting stored
# frames from their stored dofs: max|delta| 1.5e-5 on the image (float32
# CUDA-vs-CPU rounding) and ZERO mask mismatches.
# ---------------------------------------------------------------------------

# Acceptance test from camus_1.py: keep a pose only if this fraction of the cut
# slice is foreground. Rejecting nothing would feed the model empty frames from
# poses that shot past the volume edge.
RESAMPLE_DELTA = 0.35
RESAMPLE_LOWER_BOUND = 0.0
RESAMPLE_MAX_TRIALS = 500


class _SliceCutter:
    """Cut a 2D frame out of a volume at a 6-DoF pose. CPU, dataloader-worker safe."""

    def __init__(self, hw, mode="bilinear"):
        self.mode = mode
        self.slice_size = [1] + list(hw)
        v = [torch.linspace(-0.5 * (s - 1), 0.5 * (s - 1), steps=s)
             for s in self.slice_size]
        g = torch.meshgrid(v, indexing="ij")
        self.grid = torch.stack(
            [g[2], g[1], g[0], torch.ones_like(g[0])], 0).view(4, -1).float()

    @staticmethod
    def dof2mat(dof):
        rad = torch.deg2rad(dof[:, 3:])
        ai, aj, ak = rad[:, 0], rad[:, 1], rad[:, 2]
        si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc, cs, sc, ss = ci * ck, ci * sk, si * ck, si * sk
        M = torch.eye(4, dtype=dof.dtype).repeat(dof.shape[0], 1, 1)
        M[:, 0, 0] = cj * ck; M[:, 0, 1] = sj * sc - cs; M[:, 0, 2] = sj * cc + ss
        M[:, 1, 0] = cj * sk; M[:, 1, 1] = sj * ss + cc; M[:, 1, 2] = sj * cs - sc
        M[:, 2, 0] = -sj;     M[:, 2, 1] = cj * si;      M[:, 2, 2] = cj * ci
        M[:, :3, 3] = dof[:, :3]
        return M

    def __call__(self, vol, dof):
        """vol (1,1,D,H,W), dof (1,6) -> (H_s, W_s) tensor."""
        new = (self.dof2mat(dof) @ self.grid)[:, :3]
        dhw = vol.shape[2:]
        for i in range(3):
            new[:, i] = 2 * ((new[:, i] + 0.5 * (dhw[2 - i] - 1)) / (dhw[2 - i] - 1) - 0.5)
        new = new.permute(0, 2, 1).contiguous().view(
            vol.shape[0], *([1] + self.slice_size[1:]), 3)
        return torch.nn.functional.grid_sample(
            vol, new, align_corners=True, mode=self.mode,
            padding_mode="zeros")[0, 0, 0]


class PoolDataset(Dataset):
    """Pool-format dataset yielding (vol, frame, dof, seg); see module docstring.

    Args:
        dataset:    'CAMUS' | 'ProReg'
        split:      'train' | 'val' | 'test'
        size:       CAMUS in-plane resolution (128/192/384); None for ProReg
        pm:         DoF magnitude of the sampled frames (+-pm mm/deg)
        fold:       k-fold index (ProReg manifests; CAMUS uses official splits)
        frame_mode: 'expand'  -> one sample per (file, frame)   [default]
                    'random'  -> one sample per file, random frame each epoch
                    'all'     -> one sample per file, frames stacked (N,H,W)
        files:      explicit file list, bypassing manifest resolution
        cache_size: number of decoded pkl files to keep in memory per worker
        return_vol_mask: append the VOLUME mask (1,Z,Y,X) as a 5th element.
                    Off by default so the training contract is unchanged;
                    evaluate.py turns it on to resample it at the predicted
                    pose for Dice/HD95.
        resample:   draw a FRESH pose per access and cut the slice on the fly
                    instead of returning the stored frame. Training only —
                    passing it for val/test would make the benchmark
                    non-deterministic, so it is rejected there. Epoch length is
                    unchanged (still frames_per_file samples per volume); only
                    the poses become continuous instead of a fixed set of 4.
    """

    def __init__(self, dataset="CAMUS", split="train", size=None, pm=10, fold=0,
                 folds=None, frame_mode="expand", data_root=None,
                 files=None, cache_size=16, seed=None, return_vol_mask=False,
                 resample=False):
        if frame_mode not in ("expand", "random", "all"):
            raise ValueError(f"frame_mode must be expand|random|all, got '{frame_mode}'")
        self.dataset = dataset
        self.split = split
        self.size = _check_size(dataset, size)
        self.pm = int(pm)
        self.fold = fold
        self.folds = folds
        self.frame_mode = frame_mode
        self.return_vol_mask = bool(return_vol_mask)
        self.spacing_mm = spacing_mm(dataset, size)

        self.resample = bool(resample)
        if self.resample:
            if split != "train":
                raise ValueError(
                    f"resample=True is training-only; split='{split}' must keep "
                    f"its stored frames or the benchmark stops being "
                    f"deterministic across runs and models.")
            if frame_mode == "all":
                raise ValueError("resample=True does not support frame_mode='all'")
        self._cutters = None            # built lazily, per worker

        self.files = list(files) if files is not None else pool_files(
            dataset, split, size, pm, fold, folds, data_root)
        self._cache = _LRU(cache_size)
        self._rng = np.random.RandomState(seed if seed is not None else 0)
        self._vol_shape = self._slice_shape = None

        # Frame count is fixed at preprocessing time; read it once from the
        # first file and verify per load rather than opening all of them here.
        self.frames_per_file = int(_load_pool_pkl(self.files[0])[2].shape[0])

        if frame_mode == "expand":
            self.index = [(fi, k) for fi in range(len(self.files))
                          for k in range(self.frames_per_file)]
        else:
            self.index = [(fi, None) for fi in range(len(self.files))]

    def __len__(self):
        return len(self.index)

    @property
    def vol_shape(self):
        """(D,H,W) of the volumes — models sized from this instead of a
        hardcoded 128 keep working when --size changes."""
        if self._vol_shape is None:
            self._vol_shape = tuple(int(x) for x in self._load(0)[0].shape)
        return self._vol_shape

    @property
    def slice_shape(self):
        """(H,W) of the frames. Equal to the volume's in-plane size for both
        CAMUS and ProReg pools (no separate frame crop)."""
        if self._slice_shape is None:
            self._slice_shape = tuple(int(x) for x in self._load(0)[2].shape[1:])
        return self._slice_shape

    def describe(self):
        n_frames = "all" if self.frame_mode == "all" else self.frames_per_file
        size_str = f"{self.size}px" if self.size else "native"
        where = ("official split" if self.folds is None
                 else f"fold {self.fold}/{self.folds}")
        how = (f" | RESAMPLED: fresh pose per access (stored frames unused)"
               if self.resample else "")
        return (f"{self.dataset}[{self.split}] {size_str} pm{self.pm} "
                f"{where} | {len(self.files)} volumes x {n_frames} frames "
                f"= {len(self)} samples | spacing {self.spacing_mm:.4f}mm{how}")

    def _load(self, file_idx):
        return self._cache.get_or_load(self.files[file_idx], _load_pool_pkl)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.index[idx]
        vol, vol_msk, slices, slice_masks, dofs = self._load(file_idx)

        if slices.shape[0] != self.frames_per_file:
            raise ValueError(
                f"{self.files[file_idx]}: has {slices.shape[0]} frames but the "
                f"pool's first file has {self.frames_per_file} — mixed pools?")

        vol_t = torch.from_numpy(np.ascontiguousarray(vol)[None]).float()  # (1,Z,Y,X)
        extra = ()
        if self.return_vol_mask:
            extra = (torch.from_numpy(np.ascontiguousarray(vol_msk)[None]).float(),)

        if self.frame_mode == "all":
            frame_t = torch.from_numpy(np.ascontiguousarray(slices)).float()       # (N,H,W)
            seg_t = torch.from_numpy(np.ascontiguousarray(slice_masks)).float()    # (N,H,W)
            dof_t = torch.from_numpy(np.ascontiguousarray(dofs)).float()           # (N,6)
            return (vol_t, frame_t, dof_t, seg_t) + extra

        if self.resample:
            frame_t, seg_t, dof_t = self._cut_random(vol_t, vol_msk, slices.shape[1:])
            return (vol_t, frame_t, dof_t, seg_t) + extra

        if frame_idx is None:                       # 'random'
            frame_idx = int(self._rng.randint(self.frames_per_file))

        frame_t = torch.from_numpy(np.ascontiguousarray(slices[frame_idx])[None]).float()
        seg_t = torch.from_numpy(np.ascontiguousarray(slice_masks[frame_idx])[None]).float()
        dof_t = torch.from_numpy(np.ascontiguousarray(dofs[frame_idx])).float()
        return (vol_t, frame_t, dof_t, seg_t) + extra

    def _cut_random(self, vol_t, vol_msk, hw):
        """Draw a fresh pose and cut (frame, seg, dof) — the preprocessing recipe.

        Uses the ambient torch RNG rather than self._rng: PyTorch reseeds every
        dataloader worker differently at each epoch, so poses are fresh every
        epoch without any manual epoch bookkeeping. self._rng is seeded once at
        construction and would repeat itself forever.
        """
        if self._cutters is None:
            self._cutters = (_SliceCutter(hw, "bilinear"), _SliceCutter(hw, "nearest"))
        cut_img, cut_msk = self._cutters

        vol = vol_t.unsqueeze(0)                                     # (1,1,D,H,W)
        msk = torch.from_numpy(np.ascontiguousarray(vol_msk)).float()[None, None]

        for _ in range(RESAMPLE_MAX_TRIALS):
            dof = (torch.rand(1, 6) - 0.5) * 2.0 * float(self.pm)
            frame = cut_img(vol, dof)
            if (frame > RESAMPLE_LOWER_BOUND).float().mean() > RESAMPLE_DELTA:
                seg = cut_msk(msk, dof).round()
                return frame[None], seg[None], dof[0]

        # Exhausting 500 trials means the pose range cannot produce a valid view
        # of this volume — a data problem, not something to paper over with a
        # silently-degenerate sample.
        raise RuntimeError(
            f"no pose passed the {RESAMPLE_DELTA:.0%} foreground test in "
            f"{RESAMPLE_MAX_TRIALS} trials (pm={self.pm}, vol={tuple(vol.shape[2:])})")






def _case_index(path):
    """Numeric handle for a pool file — must be a
    number, not the string case id."""
    digits = re.findall(r"\d+", osp.basename(path))
    return int(digits[0]) if digits else 0


def dof2mat_np(dof6):
    """[tx,ty,tz,rx,ry,rz] (mm, degrees) -> 4x4 rigid matrix.

    Same Euler construction (sxyz) as data_utils.FrameRigidTransformer.dof2mat
    and the preprocessing scripts, so GT matrices are consistent everywhere.
    """
    dof6 = np.asarray(dof6, dtype=np.float64)
    ai, aj, ak = np.deg2rad(dof6[3]), np.deg2rad(dof6[4]), np.deg2rad(dof6[5])
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.eye(4, dtype=np.float64)
    M[0, 0] = cj * ck; M[0, 1] = sj * sc - cs; M[0, 2] = sj * cc + ss
    M[1, 0] = cj * sk; M[1, 1] = sj * ss + cc; M[1, 2] = sj * cs - sc
    M[2, 0] = -sj;     M[2, 1] = cj * si;      M[2, 2] = cj * ci
    M[:3, 3] = dof6[:3]
    return M


# ---------------------------------------------------------------------------
# Config / CLI glue — the ONE place that turns args+config into datasets
# ---------------------------------------------------------------------------
def add_data_args(parser):
    """Attach the standard data flags. Every train/eval entry point uses these,
    so `--pm 20 --size 384` means the same thing for every model.

    """
    taken = {s for a in parser._actions for s in a.option_strings}

    def add(*flags, **kw):
        if not any(f in taken for f in flags):
            parser.add_argument(*flags, **kw)

    add("--dataset", default=None, choices=list(KNOWN_DATASETS),
        help="CAMUS | ProReg (default: from config, else CAMUS)")
    add("--size", type=int, default=None,
        help="CAMUS in-plane resolution: 128|192|384 (ProReg: n/a)")
    add("--pm", type=int, default=None,
        help="DoF magnitude of the sampled frames, +-pm mm/deg (default 10)")
    add("--fold", default=None,
        help='which fold to run: an int 0..K-1, or "all" to train them all')
    add("--frame-mode", dest="frame_mode", default=None,
        choices=["expand", "random", "all"], help="expand (default) | random | all")
    add("--data-root", dest="data_root", default=None,
        help=f"dataset base dir (default {DEFAULT_DATA_ROOT})")
    add("--folds", type=int, default=None,
        help="total number of folds K (null/omitted = the official split)")
    return parser


def resolve_data_cfg(args=None, cfg=None, default_dataset="CAMUS"):
    """Merge CLI args over the config's `data` block over the defaults.

    Precedence: CLI flag > config > default. An unset flag (None) never
    clobbers the config, so a sweep can override one field without touching
    the JSON.

    `data.dataset` is shared with the OASIS/synthRAD loader (datasets/loader.py)
    which puts its own names there, so it is honoured here ONLY when it names a
    pool dataset; anything else leaves `default_dataset` in place. An explicit
    --dataset always wins.

    `size` is per-dataset. Switching dataset on the CLI (`--dataset ProReg`
    against a CAMUS config) drops a `size` that came from the config, since it
    described the OTHER dataset; an explicit `--size` on a dataset that has no
    resolutions is still an error, because that one is a real mistake.
    """
    out = {"dataset": default_dataset, "size": None, "pm": 10, "fold": 0,
           "frame_mode": "expand", "data_root": None, "folds": None,
           "resample_train": False}
    if cfg:
        d = dict(cfg.get("data", {}))
        for k in out:
            v = d.get(k)
            if v is None:
                continue
            if k == "dataset" and v not in KNOWN_DATASETS:
                continue                      # belongs to the other loader
            out[k] = v
    cli_size = getattr(args, "size", None) if args else None
    if args:
        for k in out:
            v = getattr(args, k, None)
            if v is not None:
                out[k] = v
    if _spec(out["dataset"])["default_size"] is None and cli_size is None:
        out["size"] = None                    # stale config size for another dataset
    out["size"] = _check_size(out["dataset"], out["size"])
    return out


def build_dataset(split, args=None, cfg=None, dataset_cls=PoolDataset,
                  default_dataset="CAMUS", **overrides):
    """The single entry point train/eval scripts call."""
    dc = resolve_data_cfg(args, cfg, default_dataset)
    dc.update(overrides)
    return dataset_cls(dataset=dc["dataset"], split=split, size=dc["size"],
                       pm=dc["pm"], fold=dc["fold"], folds=dc.get("folds"),
                       frame_mode=dc["frame_mode"], data_root=dc["data_root"])


def cli_datasets(*splits, default_dataset="CAMUS", dataset_cls=PoolDataset,
                 description=None, argv=None, **overrides):
    """Parse the standard data flags off the command line and build one dataset
    per split.

    """
    ap = argparse.ArgumentParser(description=description)
    add_data_args(ap)
    args, _unknown = ap.parse_known_args(argv)
    out = [build_dataset(s, args=args, dataset_cls=dataset_cls,
                         default_dataset=default_dataset, **overrides)
           for s in splits]
    for ds in out:
        print(f"[data] {ds.describe()}")
    return out if len(out) > 1 else out[0]


if __name__ == "__main__":
    import argparse
    ap = add_data_args(argparse.ArgumentParser(description="Inspect a pool split."))
    ap.add_argument("--split", default="train", choices=SPLITS)
    a = ap.parse_args()
    ds = build_dataset(a.split, args=a)
    print(ds.describe())
    vol, frame, dof, seg = ds[0]
    print(f"  vol   {tuple(vol.shape)}  [{vol.min():.3f}, {vol.max():.3f}]")
    print(f"  frame {tuple(frame.shape)}  [{frame.min():.3f}, {frame.max():.3f}]")
    print(f"  dof   {tuple(dof.shape)}  {np.round(dof.numpy(), 2)}")
    print(f"  seg   {tuple(seg.shape)}  labels={sorted(set(seg.flatten().tolist()))[:6]}")
