#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_manifests.py — write the split manifests a pool directory is missing.

CAMUS ships official splits (camus_1.py copies them into
manifests/official_{train,val,test}.txt), so only ProReg needs generating.

Manifests are k-fold over CASE ids. Splitting by case (not by file) is the
point: a case can contribute several sub-volumes cut from the same scan, so a
file-level split would put the same anatomy in train and test.

Every fold is a THREE-way split in a 7:1:2 ratio (train:val:test): the cases
are partitioned into ten chunks, and fold k takes two as test, one as val and
the remaining seven as train. The three sets are disjoint and every case is in
the test set exactly once across the k folds.

The case order is shuffled once with a fixed seed, so re-running reproduces
byte-identical manifests.

    python -m data.make_manifests --dataset ProReg           # 5 folds
    python -m data.make_manifests --dataset ProReg --folds 5 --seed 42
"""

import argparse
import os
import os.path as osp
import re

import numpy as np

from .pool_dataset import (DEFAULT_DATA_ROOT, dataset_dir, pool_dir, _spec,
                           FOLD_DIR, FOLD_PATTERN)


def collect_cases(dataset, size=None, pm=10, data_root=None):
    """Unique case IDs present in the pool (each contributes >=1 pool file)."""
    pdir = pool_dir(dataset, size, pm, data_root)
    if not osp.isdir(pdir):
        raise FileNotFoundError(f"no pool directory at {pdir}")
    case_re = _spec(dataset)["case_re"]
    cases = set()
    for fn in os.listdir(pdir):
        if fn.endswith(".pkl"):
            m = case_re.match(fn)
            if m:
                cases.add(m.group(1))
    if not cases:
        raise RuntimeError(f"no case IDs recognised in {pdir}")
    return sorted(cases)


def kfold_splits(cases, folds=5, seed=42):
    """-> list of (train, val, test) case-id lists, one per fold.

    Every fold is a THREE-way split in a 7:1:2 ratio (train:val:test). The
    cases are partitioned into 5*folds chunks; fold k takes 2*folds of them as
    test, folds as val, and the remaining 7*folds as train. Splitting by case
    (never by file) keeps the two sub-volumes of one scan on the same side, and
    every case appears in the test set exactly once across the k folds.

    The order is shuffled once with a fixed seed, so re-running reproduces
    byte-identical manifests.
    """
    order = list(cases)
    np.random.RandomState(seed).shuffle(order)
    # ten chunks: 2 test + 1 val + 7 train = 7:1:2
    n_chunks = 10
    if len(order) < n_chunks:
        raise ValueError(f"need at least {n_chunks} cases for a 7:1:2 split, "
                         f"got {len(order)}")
    chunks = [list(c) for c in np.array_split(np.array(order, dtype=object), n_chunks)]

    out = []
    for k in range(folds):
        # disjoint test pairs across the folds, so each case is tested once
        t0, t1 = (2 * k) % n_chunks, (2 * k + 1) % n_chunks
        v = (2 * k + 2) % n_chunks
        test = chunks[t0] + chunks[t1]
        val = chunks[v]
        train = [c for i, ch in enumerate(chunks) if i not in (t0, t1, v)
                 for c in ch]
        out.append((sorted(train), sorted(val), sorted(test)))
    return out


def write_manifests(dataset, size=None, pm=10, folds=5, seed=42, data_root=None):
    cases = collect_cases(dataset, size, pm, data_root)
    # K lives in the path: a 5-fold and a 10-fold set are different splits and
    # must not overwrite each other.
    mdir = osp.join(dataset_dir(dataset, size, data_root), "manifests",
                    FOLD_DIR.format(folds=int(folds)))
    os.makedirs(mdir, exist_ok=True)

    print(f"{dataset}: {len(cases)} cases -> {folds}-fold manifests in {mdir}")
    for k, (train, val, test) in enumerate(kfold_splits(cases, folds, seed)):
        for split, ids in (("train", train), ("val", val), ("test", test)):
            path = osp.join(mdir, FOLD_PATTERN.format(fold=k, split=split))
            with open(path, "w") as f:
                f.write("\n".join(ids) + "\n")
        print(f"  fold{k}: train={len(train)} val={len(val)} test={len(test)}")

    # Cheap invariants: the three sets must be disjoint per fold, and every
    # case must land in test exactly once across all folds.
    seen_test = []
    for k, (train, val, test) in enumerate(kfold_splits(cases, folds, seed)):
        assert not (set(train) & set(val)), f"fold{k}: train/val overlap"
        assert not (set(train) & set(test)), f"fold{k}: train/test overlap"
        assert not (set(val) & set(test)), f"fold{k}: val/test overlap"
        seen_test += test
    assert sorted(seen_test) == sorted(cases), "test folds do not cover every case once"
    print("  checks passed: splits disjoint per fold, test folds cover all cases once")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ProReg")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--pm", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", dest="data_root", default=DEFAULT_DATA_ROOT)
    a = ap.parse_args()
    write_manifests(a.dataset, a.size, a.pm, a.folds, a.seed, a.data_root)


if __name__ == "__main__":
    main()
