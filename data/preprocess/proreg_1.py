#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds ONE flat pool of self-contained per-(case, sub-volume) pkl files for
the prostate US self-registration task (mirrors the CAMUS pool design, with
subA/subB playing the role CAMUS gives 2CH/4CH).

Each pool file bundles one 40-deep sub-volume, its (dummy) mask, and
NUM_SLICES_PER_SUBVOL fixed-pose frame samples (frame, frame-mask, dof)
drawn from it -- the sub-volume is stored ONCE per file, not once per frame.

Unlike CAMUS, this is a genuine 3D anatomical volume (real spatial depth,
not a "time-as-depth" pseudo-volume), so out-of-plane rotation does not
suffer the Z-clipping problem found for CAMUS -- no resolution-dependent DoF
safety margin is needed here.

Reads preprocess_0's cache (Prostate40x80x96/processed_pkl_normalized, built
by `proreg_0.py`, which must run first).
"""

import argparse
import os, os.path as osp, pickle, re
from glob import glob
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
_ap = argparse.ArgumentParser()
_ap.add_argument("--pm", type=int, default=10,
                 help="uniform DoF magnitude of the sampled frames (+-pm mm/deg); "
                      "output goes to ProReg/pool_pm{pm}")
_args = _ap.parse_args()

PROC_PKL = f"{DATA_ROOT}/ProReg/processed_pkl_normalized"
OUT_ROOT = f"{DATA_ROOT}/ProReg"
OUT_POOL = osp.join(OUT_ROOT, f"pool_pm{_args.pm}")
os.makedirs(OUT_POOL, exist_ok=True)

SUB_DEPTH = 40
# (H,W) of the stored volume = the ACQUISITION plane (Z,Y) after proreg_0's
# crop. The name is legacy: H is native Z, W is native Y, and the stacking
# axis D runs along native X. Must match proreg_0's Z_KEEP/Y_KEEP.
INPLANE_SHAPE_YX = (64, 64)          # must match preprocess_0
SLICE_HW = INPLANE_SHAPE_YX          # frame is the SAME in-plane size as the volume
SUB_TAGS = ("subA", "subB")          # must match preprocess_0's naming

NUM_SLICES_PER_SUBVOL = 4
SCALER = np.array([_args.pm] * 6, dtype=np.float32)  # [tx,ty,tz,rx,ry,rz] mm/deg, uniform

DELTA, LOWER_BOUND = 0.7, 0.0  # verified empirically -- re-verify empirically for this data before a full batch
SEED_OFFSET = 42


def save_case_pkl(path_out, vol_zyx, msk_zyx, slices, slice_masks, dofs):
    payload = (
        vol_zyx.astype(np.float32, copy=False),                # (Z,Y,X)
        msk_zyx.astype(np.uint8,  copy=False),                 # (Z,Y,X)
        np.stack(slices).astype(np.float32, copy=False),       # (N,H,W)
        np.stack(slice_masks).astype(np.uint8, copy=False),    # (N,H,W)
        np.stack(dofs).astype(np.float32, copy=False),         # (N,6)
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
        # padding_mode='zeros': out-of-bounds -> no data (black), not a smeared border replicate
        # (see the CAMUS pipeline's stripe-artifact finding for why 'border' is avoided).
        return F.grid_sample(vol, new_locs, align_corners=True, mode=self.mode,
                             padding_mode='zeros')


def generate_fixed_slices_for_case(case: str, out_dir: str, num_slices: int, seed_offset: int):
    for tag in SUB_TAGS:
        vol_pkl_path = osp.join(
            PROC_PKL, f"{case}_{tag}_z{SUB_DEPTH}y{INPLANE_SHAPE_YX[0]}x{INPLANE_SHAPE_YX[1]}.pkl"
        )
        if not osp.isfile(vol_pkl_path):
            continue

        with open(vol_pkl_path, "rb") as f:
            vol_zyx, msk_zyx, _meta = pickle.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        frt_img  = FrameRigidTransformer(SLICE_HW, mode='bilinear').to(device)
        frt_mask = FrameRigidTransformer(SLICE_HW, mode='nearest').to(device)

        vol_th = torch.from_numpy(vol_zyx).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        msk_th = torch.from_numpy(msk_zyx.astype(np.float32)).to(device=device).unsqueeze(0).unsqueeze(0)

        base = 13 if tag == "subA" else 17
        seed = (abs(hash(case)) % (2**31 - 1)) * base + seed_offset
        seed = seed & 0x7fffffff
        rng = np.random.RandomState(seed)

        slices, slice_masks, dofs = [], [], []
        got, trials = 0, 0
        while got < num_slices and trials < 500:
            trials += 1
            dof_np = (rng.rand(6).astype(np.float32) - 0.5) * 2.0 * SCALER
            dof = torch.from_numpy(dof_np[None, :]).to(device)

            slc  = frt_img(vol_th,  dof)[0, 0, 0].detach().cpu().numpy()
            slcm = frt_mask(msk_th, dof)[0, 0, 0].detach().cpu().numpy().round().astype(np.uint8)

            score = (slc > LOWER_BOUND).mean()
            if score <= DELTA:
                continue

            slices.append(slc)
            slice_masks.append(slcm)
            dofs.append(dof_np)
            got += 1

        if got == 0:
            continue

        out_name = f"{case}_{tag}_{SUB_DEPTH}x{INPLANE_SHAPE_YX[0]}x{INPLANE_SHAPE_YX[1]}.pkl"
        out_path = osp.join(out_dir, out_name)
        save_case_pkl(out_path, vol_zyx, msk_zyx, slices, slice_masks, dofs)


def all_cases():
    pattern = re.compile(r"(case\d+)_sub[A-Z]_z")
    ids = set()
    for fn in os.listdir(PROC_PKL):
        m = pattern.match(fn)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def check_volume_cache():
    if not osp.isdir(PROC_PKL) or not os.listdir(PROC_PKL):
        raise FileNotFoundError(
            f"No preprocess_0 volume cache found in {PROC_PKL}. Run `python proreg_0.py` first."
        )


def build_pool():
    cases = all_cases()
    print(f"Generating {NUM_SLICES_PER_SUBVOL} fixed frames per sub-volume for {len(cases)} cases -> {OUT_POOL}")
    for case in tqdm(cases, desc="Generate pool slices"):
        generate_fixed_slices_for_case(case, OUT_POOL, NUM_SLICES_PER_SUBVOL, seed_offset=SEED_OFFSET)


def main():
    print(f"sub-volume shape (Z,Y,X)=({SUB_DEPTH},{INPLANE_SHAPE_YX[0]},{INPLANE_SHAPE_YX[1]}), "
          f"frame (H,W)={SLICE_HW}, DoF=+-{SCALER.tolist()}")
    check_volume_cache()
    build_pool()
    print("\nDone.")
    print(f"Pool -> {OUT_POOL}")


if __name__ == "__main__":
    main()
