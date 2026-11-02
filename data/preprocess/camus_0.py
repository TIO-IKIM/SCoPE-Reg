#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the per-patient/view volume+mask cache (one pkl per patient/view,
containing BOTH the resampled image and its segmentation mask).

Field-of-view is held CONSTANT across resolutions: the reference is
128px @ 0.616mm/px (FOV = 128*0.616 = 78.848mm). Any other --size just
samples that SAME physical window more finely (spacing = FOV/size), so the
anatomy always fills the frame the same way regardless of resolution --
only the level of detail changes. Going below the native ~0.308mm spacing
stops adding real information (pure interpolation past that point).
"""

import argparse
import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

# Dataset location. Override with the SVR_DATA_ROOT environment variable so
# this script runs unmodified on another machine.
DATA_ROOT = os.environ.get("SVR_DATA_ROOT", "./datasets")


_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--size", type=int, default=128,
                         help="in-plane pixel resolution (X and Y are both size); "
                              "output goes to $SVR_DATA_ROOT/CAMUS{size}/processed_pkl_normalized")
_args = _arg_parser.parse_args()

REFERENCE_SIZE = 128
REFERENCE_SPACING_MM = 0.616
REFERENCE_FOV_MM = REFERENCE_SIZE * REFERENCE_SPACING_MM  # 78.848mm, held constant across resolutions

ROOT = f"{DATA_ROOT}/database_nifti/"
OUT_ROOT = f"{DATA_ROOT}/CAMUS{_args.size}/processed_pkl_normalized"
os.makedirs(OUT_ROOT, exist_ok=True)

TARGET_SIZE_ZXY = (32, _args.size, _args.size)          # (Z, X, Y)
_spacing_mm = REFERENCE_FOV_MM / _args.size
TARGET_SPACING = (_spacing_mm, _spacing_mm, _spacing_mm)  # (sx, sy, sz) in mm (sitk uses (x,y,z))
DICOM_ORIENT = "RAI"

def reorient_to(img, orient_code=DICOM_ORIENT):
    return sitk.DICOMOrient(img, orient_code)

def physical_center(img: sitk.Image):
    size = np.array(list(img.GetSize()), dtype=np.float64)        # (x,y,z)
    spacing = np.array(list(img.GetSpacing()), dtype=np.float64)  # (x,y,z)
    origin = np.array(list(img.GetOrigin()), dtype=np.float64)
    direction = np.array(list(img.GetDirection()), dtype=np.float64).reshape(3,3)
    center_index = (size - 1.0) / 2.0
    return origin + direction.dot(center_index * spacing)

def resample_iso_centered(img: sitk.Image,
                          out_spacing=(1.0,1.0,1.0),
                          out_size_xyz=(192,192,32),
                          interp=sitk.sitkLinear) -> sitk.Image:
    img = reorient_to(img, DICOM_ORIENT)

    out_direction = img.GetDirection()

    C = physical_center(img)
    out_size = np.array(out_size_xyz, dtype=np.int64)             # (x,y,z)
    out_spacing = np.array(out_spacing, dtype=np.float64)         # (x,y,z)
    dir_mat = np.array(out_direction).reshape(3,3)

    half_idx = (out_size - 1.0) / 2.0
    out_origin = C - dir_mat.dot(half_idx * out_spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(tuple(out_size.tolist()))
    resampler.SetOutputSpacing(tuple(out_spacing.tolist()))
    resampler.SetOutputDirection(tuple(out_direction))
    resampler.SetOutputOrigin(tuple(out_origin.tolist()))
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()) if img.GetPixelIDValue()!=sitk.sitkUInt8 else 0.0)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))

    out = resampler.Execute(img)
    return out

def save_pkl(vol_zyx: np.ndarray, msk_zyx: np.ndarray, meta: dict, out_path: str):
    vol_zyx = vol_zyx.astype(np.float32, copy=False)
    msk_zyx = msk_zyx.astype(np.uint8, copy=False)
    with open(out_path, 'wb') as f:
        pickle.dump((vol_zyx, msk_zyx, meta), f, protocol=pickle.HIGHEST_PROTOCOL)

def process_case(case_id: int):
    pid = f"patient{case_id:04d}"
    case_dir = osp.join(ROOT, pid)
    if not osp.isdir(case_dir):
        return False, f"{pid}: folder missing"

    in_files = [
        (f"{pid}_2CH_half_sequence.nii.gz", f"{pid}_2CH_half_sequence_gt.nii.gz", "2CH"),
        (f"{pid}_4CH_half_sequence.nii.gz", f"{pid}_4CH_half_sequence_gt.nii.gz", "4CH"),
    ]

    for fname, mask_fname, view in in_files:
        in_path = osp.join(case_dir, fname)
        mask_path = osp.join(case_dir, mask_fname)
        if not (osp.isfile(in_path) and osp.isfile(mask_path)):
            continue

        img = sitk.ReadImage(in_path)
        msk = sitk.ReadImage(mask_path)
        img_res = resample_iso_centered(
            img,
            out_spacing=TARGET_SPACING,
            out_size_xyz=(TARGET_SIZE_ZXY[1], TARGET_SIZE_ZXY[2], TARGET_SIZE_ZXY[0]),
            interp=sitk.sitkLinear
        )
        msk_res = resample_iso_centered(
            msk,
            out_spacing=TARGET_SPACING,
            out_size_xyz=(TARGET_SIZE_ZXY[1], TARGET_SIZE_ZXY[2], TARGET_SIZE_ZXY[0]),
            interp=sitk.sitkNearestNeighbor
        )

        arr_zyx = sitk.GetArrayFromImage(img_res)  # (Z, Y, X)
        msk_zyx = sitk.GetArrayFromImage(msk_res)  # (Z, Y, X)

        if arr_zyx.dtype != np.float32:
            arr_zyx = arr_zyx.astype(np.float32)
        p1, p99 = np.percentile(arr_zyx, (1, 99))
        if p99 > p1:
            arr_zyx = np.clip((arr_zyx - p1) / (p99 - p1), 0.0, 1.0)

        z, x, y = arr_zyx.shape
        out_name = f"{pid}_{view}_half_sequence_zxy{z}x{x}x{y}.pkl"
        out_path = osp.join(OUT_ROOT, out_name)

        meta = {
            "patient_id": pid,
            "view": view,
            "orientation": DICOM_ORIENT,
            "spacing_xyz_mm": TARGET_SPACING,      # (x,y,z)
            "fov_mm": REFERENCE_FOV_MM,
            "shape_zxy": arr_zyx.shape,            # (Z,X,Y)
            "source_path": in_path,
            "mask_source_path": mask_path,
        }
        save_pkl(arr_zyx, msk_zyx, meta, out_path)

    return True, pid

def main():
    print(f"size={_args.size}px, spacing={_spacing_mm:.4f}mm/px, FOV={REFERENCE_FOV_MM:.3f}mm (fixed)")
    ok_cnt = 0
    for i in tqdm(range(1, 500+1)):
        ok, msg = process_case(i)
        if ok:
            ok_cnt += 1
        else:
            print(f"[WARN] {msg}")
    print(f"Done. processed={ok_cnt} / 500. Output -> {OUT_ROOT}")

if __name__ == "__main__":
    main()
