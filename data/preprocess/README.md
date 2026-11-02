# Dataset preprocessing

Builds the pool-format datasets that `data/pool_dataset.py` reads. Two stages
per dataset: `*_0.py` writes a volume(+mask) cache, `*_1.py` samples the fixed
frames and writes the pool. The cache is only an intermediate — it can be
deleted once the pool is built.

| script        | stage | writes                                    |
|---------------|-------|-------------------------------------------|
| `camus_0.py`  | cache | `CAMUS{size}/processed_pkl_normalized/`   |
| `camus_1.py`  | pool  | `CAMUS{size}/pool_pm{pm}/` + `manifests/` |
| `proreg_0.py` | cache | `ProReg/processed_pkl_normalized/`        |
| `proreg_1.py` | pool  | `ProReg/pool_pm{pm}/`                     |

All paths are relative to `$SVR_DATA_ROOT` (see the top-level README).

## CAMUS

Source: `$SVR_DATA_ROOT/database_nifti` and `$SVR_DATA_ROOT/database_split`.

```bash
python data/preprocess/camus_0.py --size 128          # cache
python data/preprocess/camus_1.py --size 128 --pm 10  # pool + official manifests
```

`--size` is FOV-anchored: 128px at 0.616 mm is the reference, so any other size
samples the same physical window (78.848 mm) more finely — `spacing = FOV/size`.
Anatomy keeps its apparent size across resolutions; only detail changes.

`--pm` sets the uniform DoF magnitude (+-pm mm/deg) of the sampled frames. One
cache serves every `--pm`, so build the cache once and loop the pm variants
before deleting it.

Note: out-of-plane rotation drifts the sampling plane along Z proportionally to
the pixel count, and CAMUS's Z axis is time (32 slices). At 384px that drift can
clip — accepted as added task difficulty, not a bug.

## ProReg (mu-RegPro prostate TRUS)

Source: `$SVR_DATA_ROOT/ProstateMRUS_raw/{train,val}/us_{images,labels}`.
Ultrasound only: the MR volumes are not registered to the US, so this is
single-modality self-registration.

```bash
python data/preprocess/proreg_0.py                 # cache: 2 sub-volumes per case
python data/preprocess/proreg_1.py                 # pool
python -m data.make_manifests --dataset ProReg     # 5-fold case-level splits
```

Native volumes are 0.8 mm isotropic, numpy `(Z, Y, X) = (88, 118, 81)`. The
acquisition plane — the one carrying the TRUS fan, with the probe and balloon
arcs at its apex — is the `(Z, Y)` plane, sliced along `X`. The cache therefore
stores `(D, H, W) = (X, Z, Y)`, so the axis the frames are cut along is the real
through-plane direction and every frame is a genuine ultrasound view.

The crop never resamples, so the output stays 0.8 mm isotropic:

* `Z` 88 -> 64, anchored at the probe end (the fan apex is in every frame)
* `Y` 118 -> 64, centred
* `X` 81 -> 80, centred, then split into two contiguous 40-slice sub-volumes
  (`subA`, `subB`), mirroring how CAMUS treats 2CH/4CH as two sub-cases

Both sub-volumes come from the same percentile-normalized array, so they share
an intensity scale. A 64x64 window cannot contain the whole gland (it spans up
to 80 voxels in Z and 94 in Y), so most cases are clipped — a deliberate choice
to match the 64x64 grid reported in prior work.

`us_labels/` holds a 6-channel 0/255 stack. Channel 0 is the prostate gland,
annotated in every case, and is stored as the binary mask; channels 1-5 are
small landmark regions used for TRE, not a segmentation. The mask goes through
the identical crop, transpose and split with nearest-neighbour semantics.

Splits group by case, so `subA` and `subB` of one scan never straddle a split
boundary.
