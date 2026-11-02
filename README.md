---
name: scope-reg-paper-vs-checkpoints
description: "Published SCoPE-Reg Table 1, Table 3, Table S3 and Fig. 4 differ from stored checkpoint results by round-number offsets; checked 2026-11-02; cause unknown; README must not claim those cells reproduce"
metadata:
  node_type: memory
  type: project
  originSessionId: 4e265e60-75c3-4a3e-9bc4-96b70a05d65c
  modified: 2026-11-02T10:50:44.248Z
---

Found 2026-11-02 while writing the release README. Reported to the user; the cause is unknown to me and they may hold an
explanation (e.g. the removed `~/datasets/checkpoints` archive). Re-check before repeating this as current.

**Evidence.** `make_benchmark_table.py` on `~/checkpoints` reproduces 188/200 cells of arXiv v2 Table 1 exactly. The other
12 differ by round amounts (generator -> arXiv):
- CAMUS FVR-Net: mean 7.30->4.90, TransErr 5.31->3.31, RotErr 9.00->6.36, <=3mm 1.0->3.0, Dice .922->.942, HD95 7.56->5.56, NCC .631->.731, SSIM .296->.436
- CAMUS EUReg: mean 1.04->1.24, <=3mm 99.2->97.2
- mu-RegPro EUReg: mean 2.53->2.63
- mu-RegPro Ours: max 8.31->5.95 (EUReg max 6.77, so the column's winner changes)

Fig. 4 source: Ours and DreamReg match stored values exactly; EUReg, FVR-Net and CU-Reg differ by exact offsets with the
lower digits preserved (EUReg pm25 9.0399->5.5399; FVR-Net pm15 10.0579->7.0579; std values also changed for CU-Reg).
Table 3: stored 200-epoch ablation runs give 2.28/1.00/1.06/1.14 vs published 1.91/0.68/0.72/0.89, with `max` matching on
three rows; the reference row uses the 500-epoch main run. Table S3 flow-loss rows have no stored runs.

**Fresh `evaluate.py` runs on real checkpoints (CAMUS, 5 folds, n=4000) confirm the stored values:** Ours 0.727, EUReg 1.042,
FVR-Net 7.303, Fusion_Heads 1.000. mu-RegPro could not be re-run on the original pool (gone, see [[scope-reg-release]]).

**Also:** the reported runs used `split: 2`, so checkpoint selection used the test set as validation, while the paper
describes a 7:1:2 split. The release README discloses this. `pretrained/proreg/config.yaml` says epochs 1500; the original
run config says 1000.

**How to apply:** the user instructed that the README's tables quote only the paper, so they do, labelled as such. Never
write that the checkpoints reproduce those cells. The README deliberately omits Fig. 4, Table 3, and the "41 % / 14 %"
reduction claims. Quote stored `eval_metrics.json` when you need measured values. Related: [[dreamreg-val-gt-conditioned]],
[[robustness-fig-relative-degradation]].
