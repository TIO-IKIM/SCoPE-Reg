#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds ONE flat pool of self-contained per-patient/view pkl files (one file
per patient x view) plus separate patient-ID manifest files that define
split membership.

Each pool file bundles EVERYTHING needed for that patient/view: the volume,
its segmentation mask, and NUM_SLICES_PER_VIEW fixed-pose frame samples
(frame, frame-mask, dof) drawn from it. The volume/mask are stored ONCE per
file (not once per frame) -- no redundant per-frame volume copies.

Field-of-view is held CONSTANT across resolutions (see camus_0.py):
reference is 128px @ 0.616mm/px (FOV = 78.848mm); this script reads whatever
spacing/size preprocess_0 used for --size out of its own cache, so the two
stay consistent automatically.

Why a pool instead of train/validation/testing folders: fixed 5-fold CV
(planned) just needs different groupings of the same 500 patients. Baking a
single 400/50/50 split into the directory layout would force re-running the
(slow) frame generation every time the folds change. With a pool, a fold is
nothing more than a new manifest text file of patient IDs -- the pkl files
themselves never need to be regenerated.

This step reads preprocess_0's cache (CAMUS{size}/processed_pkl_normalized,
built by `camus_0.py --size <size>`, which must run first) and
never touches the raw NIfTI database directly.

manifests/official_{train,val,test}.txt are copies of the CAMUS paper's
official patient split (subgroup_training/validation/testing.txt), included
as one valid grouping out of the box; k-fold manifests can be added later by
writing additional patient-ID lists against the same pool.
"""

import argparse
import os, os.path as osp, pickle, shutil, re
from tqdm import tqdm
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchgeometry as tgm

# Dataset location. Override with the SVR_DATA_ROOT environment variable so
# this script runs unmodified on another machine.
DATA_ROOT = os.environ.get("SVR_DATA_ROOT", "./datasets")


# -------------------
# Config
# -------------------
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--size", type=int, default=128,
                         help="in-plane pixel resolution (frame and volume are both size x size); "
                              "output goes to $SVR_DATA_ROOT/CAMUS{size}/")
_arg_parser.add_argument("--pm", type=float, default=10.0,
                         help="uniform +-magnitude applied to all 6 DoF (mm for tx,ty,tz; deg for "
                              "rx,ry,rz). Different --pm values for the same --size share the same "
                              "volume+mask cache but write to their own pool_pm{pm} subfolder, so "
                              "several DoF-range ablations can coexist without regenerating the cache.")
_args = _arg_parser.parse_args()

REFERENCE_SIZE = 128
REFERENCE_SPACING_MM = 0.616
REFERENCE_FOV_MM = REFERENCE_SIZE * REFERENCE_SPACING_MM  # 78.848mm, held constant across resolutions

SPLIT_DIR = f"{DATA_ROOT}/database_split/database_split"
# Volume+mask cache produced by `camus_0.py --size <size>` -- must be run first
# for this resolution. Both the image and its segmentation mask are cached there (3-tuple
# format); this script never touches the raw NIfTI database.
PROC_PKL  = f"{DATA_ROOT}/CAMUS{_args.size}/processed_pkl_normalized"

_pm_tag = f"{_args.pm:g}"  # e.g. 10, 15, 20, 25 (no trailing .0)
OUT_ROOT     = f"{DATA_ROOT}/CAMUS{_args.size}"
OUT_POOL     = osp.join(OUT_ROOT, f"pool_pm{_pm_tag}")  # all fixed-frame samples for this DoF range
OUT_MANIFEST = osp.join(OUT_ROOT, "manifests")    # patient-ID lists defining split/fold membership (DoF-range independent)
os.makedirs(OUT_POOL, exist_ok=True)
os.makedirs(OUT_MANIFEST, exist_ok=True)

SIZE_XYZ     = (_args.size, _args.size, 32)
SLICE_HW     = (_args.size, _args.size)
_spacing_mm  = REFERENCE_FOV_MM / _args.size  # must match preprocess_0's formula exactly

NUM_SLICES_PER_VIEW = 4                                      # same for every patient (train == val == test)
SCALER = np.array([_args.pm] * 6, dtype=np.float32)  # [tx,ty,tz,rx,ry,rz] mm/deg, uniform magnitude
DELTA, LOWER_BOUND = 0.35, 0.0  # foreground-score threshold: fraction of the resampled
                                # slice with intensity > LOWER_BOUND. Verified empirically
                                # at 192px/0.616mm (FOV 118mm): median score 0.69, min 0.43
                                # over 500 random poses -- 0.35 passes comfortably. Since FOV
                                # is now held constant across --size, this should hold at any
                                # resolution; re-verify with a dry run before a full batch.
SEED_OFFSET = 42   # single fixed offset; per-patient/view seed already varies via pid + view base

# -------------------
# Helpers
# -------------------
def save_patient_view_pkl(path_out, vol_zyx, msk_zyx, slices, slice_masks, dofs):
    payload = (
        vol_zyx.astype(np.float32, copy=False),      # (Z,Y,X)
        msk_zyx.astype(np.uint8,  copy=False),       # (Z,Y,X)
        np.stack(slices).astype(np.float32, copy=False),      # (N,H,W)
        np.stack(slice_masks).astype(np.uint8, copy=False),   # (N,H,W)
        np.stack(dofs).astype(np.float32, copy=False),        # (N,6)
    )
    with open(path_out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


class FrameRigidTransformer(nn.Module):
    def __init__(self, slice_size_hw, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.slice_size = [1] + list(slice_size_hw)   # [1,H,W]
        vectors = [torch.linspace(-0.5*(s-1), 0.5*(s-1), steps=s) for s in self.slice_size]
        grids = torch.meshgrid(vectors, indexing='ij')  # z,y,x
        grid = torch.stack([grids[2], grids[1], grids[0], torch.ones_like(grids[0])], dim=0)  # (x,y,z,1)
        self.register_buffer('grid', grid.view(4, -1).float().contiguous())

    def dof2mat(self, input_dof):
        rad = tgm.deg2rad(input_dof[:, 3:])
        ai, aj, ak = rad[:, 0], rad[:, 1], rad[:, 2]
        si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc, cs = ci*ck, ci*sk
        sc, ss = si*ck, si*sk
        M = torch.eye(4, dtype=input_dof.dtype, device=input_dof.device).repeat(input_dof.shape[0], 1, 1)
        M[:,0,0]=cj*ck; M[:,0,1]=sj*sc-cs; M[:,0,2]=sj*cc+ss
        M[:,1,0]=cj*sk; M[:,1,1]=sj*ss+cc; M[:,1,2]=sj*cs-sc
        M[:,2,0]=-sj;   M[:,2,1]=cj*si;    M[:,2,2]=cj*ci
        M[:, :3, 3] = input_dof[:, :3]
        return M

    def forward(self, vol, dof):
        mat = self.dof2mat(dof)
        new_locs = (mat @ self.grid)[:, :3]
        D,H,W = vol.shape[2], vol.shape[3], vol.shape[4]
        for i,sz in enumerate((D,H,W)):
            new_locs[:, i] = 2 * ((new_locs[:, i] + 0.5*( (D,H,W)[2-i]-1)) / ((D,H,W)[2-i]-1) - 0.5)
        new_locs = new_locs.permute(0,2,1).contiguous().view(vol.shape[0], *([1]+list(self.slice_size[1:])), 3)
        # padding_mode='zeros': out-of-bounds samples (translation/rotation pushing the
        # frame window past the volume's own array edge) must read as "no data" (black),
        # not replicate the edge column -- 'border' created a smeared diagonal stripe
        # artifact wherever the sampling window exceeded the volume's X/Y extent.
        return F.grid_sample(vol, new_locs, align_corners=True, mode=self.mode,
                             padding_mode='zeros')

def generate_fixed_slices_for_patient(pid: str, out_dir: str, num_per_view: int, seed_offset: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frt_img  = FrameRigidTransformer(SLICE_HW, mode='bilinear').to(device)
    frt_mask = FrameRigidTransformer(SLICE_HW, mode='nearest').to(device)

    for view in ("2CH", "4CH"):
        vol_pkl_name = f"{pid}_{view}_half_sequence_zxy{SIZE_XYZ[2]}x{SIZE_XYZ[0]}x{SIZE_XYZ[1]}.pkl"
        vol_pkl_path = osp.join(PROC_PKL, vol_pkl_name)
        if not osp.isfile(vol_pkl_path):
            continue

        with open(vol_pkl_path, "rb") as f:
            vol_zyx, msk_zyx, _meta = pickle.load(f)   # both already resampled+normalized by preprocess_0

        vol_th = torch.from_numpy(vol_zyx).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1,1,Z,Y,X)
        msk_th = torch.from_numpy(msk_zyx.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)

        base = 13 if view=="2CH" else 17
        seed = (int(pid[-4:]) * base + seed_offset) & 0x7fffffff
        rng = np.random.RandomState(seed)

        slices, slice_masks, dofs = [], [], []
        got, trials = 0, 0
        while got < num_per_view and trials < 500:
            trials += 1
            dof_np = (rng.rand(6).astype(np.float32) - 0.5) * 2.0 * SCALER
            dof = torch.from_numpy(dof_np[None,:]).to(device)

            slc  = frt_img(vol_th,  dof)[0,0,0].detach().cpu().numpy()
            slcm = frt_mask(msk_th, dof)[0,0,0].detach().cpu().numpy().round().astype(np.uint8)

            score = (slc > LOWER_BOUND).mean()
            if score <= DELTA:
                continue

            slices.append(slc)
            slice_masks.append(slcm)
            dofs.append(dof_np)
            got += 1

        if got == 0:
            continue

        out_name = f"{pid}_{view}_{SIZE_XYZ[2]}x{SIZE_XYZ[0]}x{SIZE_XYZ[1]}.pkl"
        out_path = osp.join(out_dir, out_name)
        save_patient_view_pkl(out_path, vol_zyx, msk_zyx, slices, slice_masks, dofs)

def all_patient_ids():
    """Every patient ID actually present in preprocess_0's cache for this --size
    (some of the official 500 may be missing from the public NIfTI download)."""
    pattern = re.compile(r"(patient\d+)_(?:2CH|4CH)_half_sequence_zxy")
    ids = set()
    for fn in os.listdir(PROC_PKL):
        m = pattern.match(fn)
        if m:
            ids.add(m.group(1))
    return sorted(ids)

def copy_official_manifests():
    """Copy the CAMUS paper's official patient split as one ready-made grouping
    over the pool. Additional manifests (e.g. 5-fold CV) can be added later as
    plain patient-ID text files -- no re-generation of pool pkls required."""
    mapping = {
        "subgroup_training.txt":   "official_train.txt",
        "subgroup_validation.txt": "official_val.txt",
        "subgroup_testing.txt":    "official_test.txt",
    }
    for src_name, dst_name in mapping.items():
        src = osp.join(SPLIT_DIR, src_name)
        dst = osp.join(OUT_MANIFEST, dst_name)
        shutil.copy2(src, dst)

def build_pool():
    ids = all_patient_ids()
    print(f"Generating {NUM_SLICES_PER_VIEW} fixed frames per view for {len(ids)} patients -> {OUT_POOL}")
    for pid in tqdm(ids, desc="Generate pool slices"):
        generate_fixed_slices_for_patient(pid, OUT_POOL, NUM_SLICES_PER_VIEW, seed_offset=SEED_OFFSET)

def check_volume_cache():
    """Fail loudly (rather than silently under-generating the pool) if
    preprocess_0 hasn't been run for this --size yet."""
    expected_prefix_suffix = f"zxy{SIZE_XYZ[2]}x{SIZE_XYZ[0]}x{SIZE_XYZ[1]}.pkl"
    if not osp.isdir(PROC_PKL) or not any(
        f.endswith(expected_prefix_suffix) for f in os.listdir(PROC_PKL)
    ):
        raise FileNotFoundError(
            f"No preprocess_0 volume cache found for size={_args.size} in {PROC_PKL}. "
            f"Run `python camus_0.py --size {_args.size}` first."
        )

def main():
    print(f"size={_args.size}px, spacing={_spacing_mm:.4f}mm/px, FOV={REFERENCE_FOV_MM:.3f}mm (fixed), "
          f"pm=+-{_args.pm} (mm/deg) -> {OUT_POOL}")
    check_volume_cache()
    copy_official_manifests()
    build_pool()
    print("\nDone.")
    print(f"Pool      -> {OUT_POOL}")
    print(f"Manifests -> {OUT_MANIFEST}")

if __name__ == "__main__":
    main()
