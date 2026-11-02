#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the per-(case, sub-volume) volume(+mask) cache for the ProstateMRUS
(mu-RegPro) TRUS volumes.

AXIS CONVENTION
---------------------------------------------------------------------------
Native volumes are sitk (x,y,z) = (81,118,88) at 0.8mm isotropic, i.e. numpy
(Z,Y,X) = (88,118,81), identical across all 73 cases.

The ACQUISITION plane -- the one carrying the TRUS fan, with the probe/balloon
arcs at its apex -- is the (Z,Y) plane, sliced along X. Verified by rendering
all three centre cuts of the raw NIfTI: only the ZY cut shows the beam
geometry; the other two show the prostate en face with no fan.

The stored volume is therefore (D,H,W) = (X, Z, Y): the slice-stacking axis D
runs along X, so every frame cut from it is a real ultrasound view rather than
a synthetic reslice across the beam direction.

CROP (no resampling -- output stays 0.8mm ISOTROPIC)
---------------------------------------------------
  Z (in-plane, vertical)   88 -> 64, anchored at the PROBE END so the fan apex
                           and the balloon arcs are in every frame.
  Y (in-plane, horizontal) 118 -> 64, centred.
  X (through-plane)        81 -> 80, centred, then split into two contiguous
                           40-slice sub-volumes ("subA", "subB"), mirroring how
                           CAMUS treats 2CH/4CH as two independent sub-cases.

This clips the prostate gland in most cases (the gland spans up to 80 voxels in
Z and 94 in Y, so a 64x64 window cannot contain it) -- a deliberate choice to
match the 64x64 grid reported in prior work. See the module tests in
data/preprocess/checks/ for the measured clipping rate.

Both sub-volumes are cut from the SAME percentile-normalized array (stats are
computed once over the full crop, before the X-split), so subA/subB stay on a
consistent intensity scale.

Segmentation: `us_labels/` holds a 6-channel 0/255 stack. Channel 0 is the
prostate GLAND (annotated in 100% of cases); channels 1-5 are small landmark
regions used for TRE, not a segmentation. Channel 0 is stored as the mask
(binary 0/1) and goes through the identical crop/transpose/split with
nearest-neighbour semantics (no interpolation, no normalisation).
"""

import os
import os.path as osp
import pickle
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# Dataset location. Override with the SVR_DATA_ROOT environment variable so
# this script runs unmodified on another machine.
DATA_ROOT = os.environ.get("SVR_DATA_ROOT", "./datasets")


ROOT = f"{DATA_ROOT}/ProstateMRUS_raw"   # {train,val}/us_{images,labels}
OUT_ROOT = f"{DATA_ROOT}/ProReg/processed_pkl_normalized"
os.makedirs(OUT_ROOT, exist_ok=True)

# --- geometry, in NATIVE (Z,Y,X) terms -------------------------------------
Z_KEEP = 64              # in-plane vertical, anchored at the probe end
Y_KEEP = 64              # in-plane horizontal, centred
X_CROP = 80              # through-plane, centred (native 81), split below
SUB_DEPTH = 40           # slices per sub-volume along X (X_CROP // SUB_DEPTH sub-volumes)
SPACING_MM = 0.8         # unchanged: this pipeline crops only, never resamples
GLAND_CHANNEL = 0        # us_labels channel holding the prostate gland

# Which Z end carries the probe. The fan apex and the bright balloon arcs sit at
# HIGH Z (the bottom of a row-0-at-top render), so we keep the LAST Z_KEEP rows.
PROBE_AT_HIGH_Z = True


def crop_axis(arr, axis, keep, anchor="centre"):
    """Crop `arr` along `axis` to `keep` samples. anchor: centre | start | end."""
    n = arr.shape[axis]
    if n < keep:
        raise ValueError(f"axis {axis} has {n} < {keep}; this pipeline never pads")
    if anchor == "centre":
        start = (n - keep) // 2
    elif anchor == "start":
        start = 0
    elif anchor == "end":
        start = n - keep
    else:
        raise ValueError(anchor)
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(start, start + keep)
    return arr[tuple(slicer)]


def percentile_normalize(arr, p1=1, p99=99):
    arr = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, (p1, p99))
    if hi > lo:
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.zeros_like(arr, dtype=np.float32)


def save_pkl(vol, msk, meta, out_path):
    with open(out_path, "wb") as f:
        pickle.dump((vol.astype(np.float32, copy=False),
                     msk.astype(np.uint8, copy=False),
                     meta), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_gland_mask(label_path):
    """Channel 0 of the 6-channel us_label stack, as a binary (Z,Y,X) mask."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    if arr.ndim != 4 or arr.shape[0] < GLAND_CHANNEL + 1:
        raise ValueError(f"{label_path}: expected (C,Z,Y,X), got {arr.shape}")
    return (arr[GLAND_CHANNEL] > 0).astype(np.uint8)


def crop_to_acquisition(arr_zyx):
    """(Z,Y,X) native -> (X, Z, Y) cropped, i.e. (D,H,W) with D along X."""
    a = crop_axis(arr_zyx, 0, Z_KEEP, "end" if PROBE_AT_HIGH_Z else "start")
    a = crop_axis(a, 1, Y_KEEP, "centre")
    a = crop_axis(a, 2, X_CROP, "centre")
    return np.transpose(a, (2, 0, 1))          # (Z,Y,X) -> (X,Z,Y)


def process_one(path):
    img = sitk.ReadImage(path)
    arr_zyx = sitk.GetArrayFromImage(img).astype(np.float32)      # native (Z,Y,X)
    msk_zyx = load_gland_mask(path.replace("us_images", "us_labels"))

    vol = crop_to_acquisition(arr_zyx)                            # (X,Z,Y)
    msk = crop_to_acquisition(msk_zyx).astype(np.uint8)

    # normalise ONCE over the full crop, before the split, so subA/subB share scale
    vol = percentile_normalize(vol, 1, 99)

    base = osp.basename(path).replace(".nii.gz", "")
    n_sub = X_CROP // SUB_DEPTH
    for i in range(n_sub):
        sl = slice(i * SUB_DEPTH, (i + 1) * SUB_DEPTH)
        sub, sub_mask = vol[sl], msk[sl]
        tag = f"sub{chr(ord('A') + i)}"
        out_path = osp.join(
            OUT_ROOT, f"{base}_{tag}_z{SUB_DEPTH}y{Z_KEEP}x{Y_KEEP}.pkl")
        meta = {
            "case": base,
            "mask": "prostate gland (us_labels channel 0), binary",
            "gland_voxels": int(sub_mask.sum()),
            "sub_volume": tag,
            "x_range_in_crop": (i * SUB_DEPTH, (i + 1) * SUB_DEPTH),
            "source_path": path,
            "orig_spacing_xyz_mm": tuple(float(x) for x in img.GetSpacing()),
            "orig_size_xyz": tuple(int(x) for x in img.GetSize()),
            "spacing_mm": SPACING_MM,       # crop-only, so native spacing holds
            "shape_dhw": tuple(int(s) for s in sub.shape),
            "axes": "D=X (through-plane, slice stacking); H=Z, W=Y (acquisition plane)",
            "z_anchor": "probe end (high Z)" if PROBE_AT_HIGH_Z else "low Z",
        }
        save_pkl(sub, sub_mask, meta, out_path)


def main():
    paths = sorted(glob(osp.join(ROOT, "*", "us_images", "*.nii.gz")))
    n_sub = X_CROP // SUB_DEPTH
    print(f"{len(paths)} cases -> {n_sub} sub-volumes/case, each "
          f"(D,H,W)=({SUB_DEPTH},{Z_KEEP},{Y_KEEP}) at {SPACING_MM}mm isotropic")
    print(f"  D along X (through-plane), H=Z, W=Y (acquisition plane, probe at "
          f"{'high' if PROBE_AT_HIGH_Z else 'low'} Z)")
    for path in tqdm(paths, desc="Processing prostate US volumes"):
        process_one(path)
    print(f"Done. Output -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
