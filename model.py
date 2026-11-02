
"""SCoPE-Reg — the model, with every block named as in the paper.

    SCoPEReg
      |- ResNetEncoder8_2D / _3D    2D slice and 3D volume encoders (fixed)
      |- CWCD                       correlation-weighted coordinate decoder
      |    |- SSVFScan              the four directional passes of SSVF
      |    |    `- DISSM            dual-input state-space module
      |    `- DepthReduction        correlation-weighted reduction over depth
      `- wkps / kps        weighted Kabsch pose solver (parameter-free)

Ablation readouts also live here: SSVF (feature fusion without the coordinate
branch), CrossAttentionFusion (the attention substitute) and the direct pose
regression heads.

One file on purpose: the model reads top to bottom against the paper figures.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref




_MAX_STAGES = 4


# ==========================================================================
# Encoders  (fixed: ResNet-8, total stride 8, C = 256)
# ==========================================================================


class BasicBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

def _check(stages, layer_strides):
    if not 1 <= stages <= _MAX_STAGES:
        raise ValueError(f"stages must be 1..{_MAX_STAGES}, got {stages}")
    if len(layer_strides) != stages:
        raise ValueError(
            f"layer_strides needs one entry per stage: this encoder has "
            f"{stages} stage(s) but got {len(layer_strides)} strides "
            f"({tuple(layer_strides)}). Expected "
            f"{stages} value(s).")

class _ResNetEncoder2D(nn.Module):
    """Shared body for the shallow 2D variants; not registered on its own.

    At stages=4, width=64 it reproduces resnet_10.ResNetEncoder2D exactly,
    including the state_dict keys (stem.*, layer1..layer4.*).
    """

    Block = BasicBlock2D
    _Conv = nn.Conv2d
    _Norm = nn.BatchNorm2d
    _Pool = nn.MaxPool2d

    def __init__(self, in_channels=1, layer_strides=(1, 2, 1), stages=3, width=64):
        super().__init__()
        _check(stages, layer_strides)
        self.stages = stages
        self.width = width
        self.out_channels = width * 2 ** (stages - 1)

        stride_prod = 4                                   # the stem
        for s in layer_strides:
            stride_prod *= s
        self.total_stride = stride_prod

        self.stem = nn.Sequential(
            self._Conv(in_channels, width, kernel_size=7, stride=2, padding=3, bias=False),
            self._Norm(width),
            nn.ReLU(inplace=True),
            self._Pool(kernel_size=3, stride=2, padding=1),
        )

        # Registered as layer1..layerN (not a Sequential) so a resnet10
        # state_dict written before this generalisation still loads.
        in_ch = width
        for i in range(stages):
            out_ch = width * 2 ** i
            setattr(self, f"layer{i + 1}",
                    self.Block(in_ch, out_ch, stride=layer_strides[i]))
            in_ch = out_ch

    def forward(self, x):
        x = self.stem(x)
        for i in range(self.stages):
            x = getattr(self, f"layer{i + 1}")(x)
        return x

class _ResNetEncoder3D(_ResNetEncoder2D):
    """Shared body for the shallow 3D variants. Ref: Hara et al. CVPR 2018."""

    Block = BasicBlock3D
    _Conv = nn.Conv3d
    _Norm = nn.BatchNorm3d
    _Pool = nn.MaxPool3d

class ResNetEncoder8_2D(_ResNetEncoder2D):
    """3 stages, width 64 -> C = 256; strides (1,2,1) after a /4 stem -> /8."""

    def __init__(self, in_channels=1, layer_strides=(1, 2, 1), width=64):
        super().__init__(in_channels, layer_strides, stages=3, width=width)
        assert self.total_stride == 8 and self.out_channels == 256

class ResNetEncoder8_3D(_ResNetEncoder3D):
    """3D counterpart of ResNetEncoder8_2D; identical schedule and width."""

    def __init__(self, in_channels=1, layer_strides=(1, 2, 1), width=64):
        super().__init__(in_channels, layer_strides, stages=3, width=width)
        assert self.total_stride == 8 and self.out_channels == 256


# ==========================================================================
# DI-SSM  — dual-input state-space module (inner block of Fig. 2)
# ==========================================================================


def selective_scan(u, delta, A, B, C, D=None, delta_softplus=True):
    """selective_scan with a CPU fallback (reference impl) for tests/debugging."""
    fn = selective_scan_fn if u.is_cuda else selective_scan_ref
    return fn(u, delta, A, B, C, D, z=None, delta_bias=None,
              delta_softplus=delta_softplus)

class DISSM(nn.Module):
    _NAMES = {
        1: "PURE_CROSS", 2: "WRITE_CROSS", 3: "DELTA_ONLY",
        4: "ADD_BDELTA", 5: "ADD_ALL", 6: "FILM_BC",
    }

    def __init__(self, C, N=16, variant=1, coord_P=3):
        super().__init__()
        if variant not in self._NAMES:
            raise ValueError(f"unknown variant {variant}; expected {sorted(self._NAMES)}")
        self.C = C
        self.N = N
        self.variant = variant
        self.coord_P = coord_P

        # --- static / structured params (feature path, unchanged) ---
        A_init = torch.arange(1, N + 1, dtype=torch.float32).repeat(C, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(C))

        # --- base selective projections (feature path, unchanged) ---
        self.linear_B = nn.Linear(C, N, bias=False)
        self.linear_C = nn.Linear(C, N, bias=False)
        self.linear_Delta = nn.Linear(C, C, bias=True)
        self._init_delta_bias(self.linear_Delta)

        # --- per-variant extras (unchanged) ---
        if variant == 4:
            self.linear_B_cross = self._zero_linear(C, N)
            self.linear_Delta_cross = self._zero_linear(C, C)
        elif variant == 5:
            self.linear_B_cross = self._zero_linear(C, N)
            self.linear_C_cross = self._zero_linear(C, N)
            self.linear_Delta_cross = self._zero_linear(C, C)
        elif variant == 6:
            self.gate_B = self._const_gate(C, N)
            self.gate_C = self._const_gate(C, N)

        # --- coordinate cross-scan params (only used when coords is passed) ---
        self.a_coord_log = nn.Parameter(torch.zeros(N))     # A = -exp(a), one shared row
        self.lin_Bc = nn.Linear(C, N, bias=False)           # keys    <- volume (xa)
        self.lin_Cc = nn.Linear(C, N, bias=False)           # queries <- slice  (xb)
        self.lin_dtc = nn.Linear(C, 1, bias=True)           # write strength <- volume
        self._init_delta_bias(self.lin_dtc)

    # ---------------- init helpers (unchanged) ----------------
    @staticmethod
    def _init_delta_bias(layer, dt_min=0.001, dt_max=0.1, dt_floor=1e-4):
        c = layer.bias.shape[0]
        dt = torch.exp(torch.rand(c) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            layer.bias.copy_(inv_dt)

    @staticmethod
    def _zero_linear(inp, out):
        lin = nn.Linear(inp, out, bias=False)
        nn.init.zeros_(lin.weight)
        return lin

    @staticmethod
    def _const_gate(inp, out, bias_val=4.0):
        lin = nn.Linear(inp, out, bias=True)
        nn.init.zeros_(lin.weight)
        nn.init.constant_(lin.bias, bias_val)
        return lin

    # ---------------- projection routing (unchanged) ----------------
    def _project(self, xa, xb):
        v = self.variant
        if v == 1:
            return self.linear_B(xb), self.linear_C(xb), self.linear_Delta(xb)
        if v == 2:
            return self.linear_B(xb), self.linear_C(xa), self.linear_Delta(xb)
        if v == 3:
            return self.linear_B(xa), self.linear_C(xa), self.linear_Delta(xb)
        if v == 4:
            return (self.linear_B(xa) + self.linear_B_cross(xb),
                    self.linear_C(xa),
                    self.linear_Delta(xa) + self.linear_Delta_cross(xb))
        if v == 5:
            return (self.linear_B(xa) + self.linear_B_cross(xb),
                    self.linear_C(xa) + self.linear_C_cross(xb),
                    self.linear_Delta(xa) + self.linear_Delta_cross(xb))
        if v == 6:
            return (self.linear_B(xa) * torch.sigmoid(self.gate_B(xb)),
                    self.linear_C(xa) * torch.sigmoid(self.gate_C(xb)),
                    self.linear_Delta(xa))
        raise ValueError(f"unknown variant {v}")

    # ---------------- forward ----------------
    def forward(self, xa, xb, coords=None):
        """
        xa, xb : (B, S, C)   xa = volume stream, xb = slice stream
        coords : (B, S, P) optional. If given, returns (y, y_coord) with
                 y_coord (B, S, P+1): P unnormalized coordinate sums plus the
                 accumulated normalizer. Do NOT divide here — sum across
                 directions/stages first, divide once at the end.
        """
        B_mat, C_mat, delta_raw = self._project(xa, xb)

        u = xa.transpose(1, 2)                        # (B, C, S)
        delta_t = delta_raw.transpose(1, 2)           # (B, C, S)
        Bm = B_mat.transpose(1, 2).contiguous()       # (B, N, S)
        Cm = C_mat.transpose(1, 2).contiguous()       # (B, N, S)
        A = -torch.exp(self.A_log.float())            # (C, N)

        y = selective_scan(u, delta_t, A, Bm, Cm, self.D.float()).transpose(1, 2)

        if coords is None:
            return y                                  # <-- unchanged behaviour

        # ---- coordinate cross-scan: slice queries, volume keys, coord values ----
        P1 = self.coord_P + 1
        ones = torch.ones_like(coords[..., :1])
        u_c = torch.cat([coords, ones], dim=-1).transpose(1, 2).contiguous()      # (B, P+1, S)

        Bc = F.softplus(self.lin_Bc(xa)).transpose(1, 2).contiguous()             # (B, N, S)
        Cc = F.softplus(self.lin_Cc(xb)).transpose(1, 2).contiguous()             # (B, N, S)
        dt = self.lin_dtc(xa).transpose(1, 2).expand(-1, P1, -1).contiguous()     # (B, P+1, S)
        A_c = (-torch.exp(self.a_coord_log.float())).unsqueeze(0) \
            .expand(P1, -1).contiguous()                                          # (P+1, N)

        y_coord = selective_scan(u_c, dt, A_c, Bc, Cc, None).transpose(1, 2)      # (B, S, P+1)
        return y, y_coord


# ==========================================================================
# SSVF  — state-space slice--volume fusion
# ==========================================================================


# DI-SSM under four spatial permutations, unflattened and summed.

class SSVFScan(nn.Module):
    def __init__(self, C, d_state=16, calc_a=True, coord_P=3):
        super().__init__()
        self.C = C
        self.calc_a = calc_a
        self.coord_P = coord_P
        if self.calc_a:
            self.fssm_a = nn.ModuleList([DISSM(C=C, N=d_state) for _ in range(4)])
        self.fssm_b = nn.ModuleList([DISSM(C=C, N=d_state, coord_P=coord_P) for _ in range(4)])

    def forward(self, Xa, Xb, coords=None):
        B, C, H, W = Xa.shape
        PC = self.coord_P + 1        # coordinate channels + normalizer

        x1_a = Xa.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x1_b = Xb.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # coordinate sequences (H-scan orientation) and (V-scan orientation)
        if coords is not None:
            c1 = coords.permute(0, 2, 3, 1).contiguous().view(B, H * W, self.coord_P)
            c3 = coords.permute(0, 3, 2, 1).contiguous().view(B, W * H, self.coord_P)
            yc_list = []

        # ---- Branch B (volume), 4 directions ----
        # dir 0: H-forward
        if coords is not None:
            y1_b_seq, yc1 = self.fssm_b[0](x1_b, x1_a, c1)
            yc_list.append(yc1.view(B, H, W, PC).permute(0, 3, 1, 2).contiguous())
        else:
            y1_b_seq = self.fssm_b[0](x1_b, x1_a)
        Y1_b = y1_b_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # dir 1: H-backward
        x2_a, x2_b = torch.flip(x1_a, [1]), torch.flip(x1_b, [1])
        if coords is not None:
            c2 = torch.flip(c1, [1])
            y2_b_seq, yc2 = self.fssm_b[1](x2_b, x2_a, c2)
            y2_b_seq = torch.flip(y2_b_seq, [1])
            yc2 = torch.flip(yc2, [1])
            yc_list.append(yc2.view(B, H, W, PC).permute(0, 3, 1, 2).contiguous())
        else:
            y2_b_seq = torch.flip(self.fssm_b[1](x2_b, x2_a), [1])
        Y2_b = y2_b_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # dir 2: V-forward
        x3_a = Xa.permute(0, 3, 2, 1).contiguous().view(B, W * H, C)
        x3_b = Xb.permute(0, 3, 2, 1).contiguous().view(B, W * H, C)
        if coords is not None:
            y3_b_seq, yc3 = self.fssm_b[2](x3_b, x3_a, c3)
            yc_list.append(yc3.view(B, W, H, PC).permute(0, 3, 2, 1).contiguous())
        else:
            y3_b_seq = self.fssm_b[2](x3_b, x3_a)
        Y3_b = y3_b_seq.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()

        # dir 3: V-backward
        x4_a, x4_b = torch.flip(x3_a, [1]), torch.flip(x3_b, [1])
        if coords is not None:
            c4 = torch.flip(c3, [1])
            y4_b_seq, yc4 = self.fssm_b[3](x4_b, x4_a, c4)
            y4_b_seq = torch.flip(y4_b_seq, [1])
            yc4 = torch.flip(yc4, [1])
            yc_list.append(yc4.view(B, W, H, PC).permute(0, 3, 2, 1).contiguous())
        else:
            y4_b_seq = torch.flip(self.fssm_b[3](x4_b, x4_a), [1])
        Y4_b = y4_b_seq.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()

        Yb = Y1_b + Y2_b + Y3_b + Y4_b

        # ---- Branch A (slice), unchanged, only if needed ----
        Ya = None
        if self.calc_a:
            y1_a = self.fssm_a[0](x1_a, x1_b).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            y2_a = torch.flip(self.fssm_a[1](x2_a, x2_b), [1]).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            y3_a = self.fssm_a[2](x3_a, x3_b).view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
            y4_a = torch.flip(self.fssm_a[3](x4_a, x4_b), [1]).view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
            Ya = y1_a + y2_a + y3_a + y4_a

        if coords is None:
            return Ya, Yb

        Ycoord = sum(yc_list)        # SUM of num/den over directions (no division here)
        return Ya, Yb, Ycoord

# Feature-only variant, used by the 'SSVF + Direct Heads' ablation arm.

class SSM(nn.Module):
    """
    Exact implementation of 'Algorithm 1: SSM Block' from the paper FusionMamba.
    Expects (B, S, C) and returns (B, S, C).
    """
    def __init__(self, C, N=16):
        super().__init__()
        self.C = C
        self.N = N
        
        # 1: A : (C, N) <- ParameterA
        A_init = torch.arange(1, N + 1, dtype=torch.float32).repeat(C, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(C))

        # 2: B : (H W, N) <- LinearB(x) -> im Code (S, N)
        self.linear_B = nn.Linear(C, N, bias=False)
        
        # 3: C : (H W, N) <- LinearC(x) -> im Code (S, N)
        self.linear_C = nn.Linear(C, N, bias=False)
        
        # 4: Delta : (H W, C) <- log(1 + exp(Linear(x) + Parameter))
        self.linear_Delta = nn.Linear(C, C, bias=True)
        dt_min, dt_max, dt_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(C) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.linear_Delta.bias.copy_(inv_dt)

    def forward(self, x):
        B_size, S, C = x.shape
        
        B_mat = self.linear_B(x)       # (B, S, N)
        C_mat = self.linear_C(x)       # (B, S, N)
        delta_raw = self.linear_Delta(x)  # (B, S, C)

        u = x.transpose(1, 2)
        delta_raw = delta_raw.transpose(1, 2)
        B_mat = B_mat.transpose(1, 2).contiguous()
        C_mat = C_mat.transpose(1, 2).contiguous()
        
        A = -torch.exp(self.A_log.float())

        y = selective_scan_fn(
            u, delta_raw, A, B_mat, C_mat, self.D.float(),
            z=None, delta_bias=None, delta_softplus=True
        )
        return y.transpose(1, 2)

class _FeatureDISSM(nn.Module):
    """
    _FeatureDISSM block with switchable cross-modality fusion variant (for ablation).
 
    Signature matches the plain _FeatureDISSM: forward(xa, xb)
        xa = sequence that PROPAGATES through the scan  -> becomes u
        xb = GUIDE / conditioning sequence (the other modality)
    In the _FeatureSSVFScan volume branch: xa = volume, xb = slice.
 
    variant:
      1  PURE_CROSS   B,C,Delta <- xb
      2  WRITE_CROSS  B,Delta <- xb ;  C <- xa                     guide controls write+timescale, target keeps readout
      3  DELTA_ONLY   Delta <- xb   ;  B,C <- xa                   gentlest: guide only gates memory/selection
      4  ADD_BDELTA   B = B(xa)+Bx(xb) ; Delta = D(xa)+Dx(xb) ; C = C(xa)   combine (recommended first try)
      5  ADD_ALL      B,C,Delta each = self(xa) + cross(xb)        full additive combine
      6  FILM_BC      B = B(xa)*sig(gB(xb)) ; C = C(xa)*sig(gC(xb)) ; Delta <- xa   multiplicative modulation
 
    Variants 4-6 initialise the cross/gate contribution to ~0 (or gate ~1),
    so the block starts as a plain self-Mamba and *learns* the fusion term.
    A,D stay static/structured (standard Mamba) in every variant.
    """
 
    _NAMES = {
        1: "PURE_CROSS", 2: "WRITE_CROSS", 3: "DELTA_ONLY",
        4: "ADD_BDELTA", 5: "ADD_ALL", 6: "FILM_BC",
    }
 
    def __init__(self, C, N=16, variant=1):
        super().__init__()
        if variant not in self._NAMES:
            raise ValueError(f"unknown variant {variant}; expected one of {sorted(self._NAMES)}")
        self.C = C
        self.N = N
        self.variant = variant
 
        # --- static / structured params (standard Mamba S4D-real init, unchanged) ---
        A_init = torch.arange(1, N + 1, dtype=torch.float32).repeat(C, 1)   # (C, N)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(C))
 
        # --- base selective projections ---
        self.linear_B = nn.Linear(C, N, bias=False)
        self.linear_C = nn.Linear(C, N, bias=False)
        self.linear_Delta = nn.Linear(C, C, bias=True)
        self._init_delta_bias(self.linear_Delta)
 
        # --- extra params per variant (built only when needed for honest param counts) ---
        if variant == 4:
            self.linear_B_cross = self._zero_linear(C, N)
            self.linear_Delta_cross = self._zero_linear(C, C)
        elif variant == 5:
            self.linear_B_cross = self._zero_linear(C, N)
            self.linear_C_cross = self._zero_linear(C, N)
            self.linear_Delta_cross = self._zero_linear(C, C)
        elif variant == 6:
            self.gate_B = self._const_gate(C, N)
            self.gate_C = self._const_gate(C, N)
 
    # ---------------- init helpers ----------------
    @staticmethod
    def _init_delta_bias(layer, dt_min=0.001, dt_max=0.1, dt_floor=1e-4):
        c = layer.bias.shape[0]
        dt = torch.exp(
            torch.rand(c) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            layer.bias.copy_(inv_dt)
 
    @staticmethod
    def _zero_linear(inp, out):
        """Cross projection whose contribution is exactly 0 at init."""
        lin = nn.Linear(inp, out, bias=False)
        nn.init.zeros_(lin.weight)
        return lin
 
    @staticmethod
    def _const_gate(inp, out, bias_val=4.0):
        """weight=0, bias>0  ->  sigmoid(gate) ~ 0.98 at init (near pass-through)."""
        lin = nn.Linear(inp, out, bias=True)
        nn.init.zeros_(lin.weight)
        nn.init.constant_(lin.bias, bias_val)
        return lin
 
    # ---------------- projection routing ----------------
    def _project(self, xa, xb):
        """Return (B_mat, C_mat, delta_raw), each (batch, S, ·), pre-softplus for delta."""
        v = self.variant
        if v == 1:  # PURE_CROSS
            return self.linear_B(xb), self.linear_C(xb), self.linear_Delta(xb)
        if v == 2:  # WRITE_CROSS
            return self.linear_B(xb), self.linear_C(xa), self.linear_Delta(xb)
        if v == 3:  # DELTA_ONLY
            return self.linear_B(xa), self.linear_C(xa), self.linear_Delta(xb)
        if v == 4:  # ADD_BDELTA
            B_mat = self.linear_B(xa) + self.linear_B_cross(xb)
            C_mat = self.linear_C(xa)
            delta = self.linear_Delta(xa) + self.linear_Delta_cross(xb)
            return B_mat, C_mat, delta
        if v == 5:  # ADD_ALL
            B_mat = self.linear_B(xa) + self.linear_B_cross(xb)
            C_mat = self.linear_C(xa) + self.linear_C_cross(xb)
            delta = self.linear_Delta(xa) + self.linear_Delta_cross(xb)
            return B_mat, C_mat, delta
        if v == 6:  # FILM_BC
            B_mat = self.linear_B(xa) * torch.sigmoid(self.gate_B(xb))
            C_mat = self.linear_C(xa) * torch.sigmoid(self.gate_C(xb))
            delta = self.linear_Delta(xa)
            return B_mat, C_mat, delta
        raise ValueError(f"unknown variant {v}")
 
    # ---------------- forward ----------------
    def forward(self, xa, xb):
        # xa, xb: (B, S, C) with S = H * W
        B_mat, C_mat, delta_raw = self._project(xa, xb)
 
        u = xa.transpose(1, 2)                       # (B, C, S) -- target that propagates
        delta_raw = delta_raw.transpose(1, 2)        # (B, C, S)
        B_mat = B_mat.transpose(1, 2).contiguous()   # (B, N, S)
        C_mat = C_mat.transpose(1, 2).contiguous()   # (B, N, S)
 
        A = -torch.exp(self.A_log.float())           # (C, N), static
 
        y = selective_scan_fn(
            u, delta_raw, A, B_mat, C_mat, self.D.float(),
            z=None, delta_bias=None, delta_softplus=True,
        )
        return y.transpose(1, 2)

class _FeatureSSVFScan(nn.Module):
    """
    Performs the 4-directional symmetrical cross-SSM scan on two 2D feature maps.
    Inputs:  Xa (B, C, H, W), Xb (B, C, H, W)
    Outputs: Ya (B, C, H, W), Yb (B, C, H, W)
    
    With calc_a=False the A branch (Ya) is skipped entirely, halving both the
    computation and the parameter count.
    """
    def __init__(self, C, d_state=16, calc_a=True):
        super().__init__()
        self.C = C
        self.calc_a = calc_a
        
        # Only built when the A branch (the slice update) is actually needed
        if self.calc_a:
            self.fssm_a = nn.ModuleList([_FeatureDISSM(C=C, N=d_state) for _ in range(4)])
            
        self.fssm_b = nn.ModuleList([_FeatureDISSM(C=C, N=d_state) for _ in range(4)])

    def forward(self, Xa, Xb):
        B, C, H, W = Xa.shape
        
        # Basis-Vektoren vorbereiten
        x1_a = Xa.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x1_b = Xb.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        # --- branch B (volume) ---
        y1_b_seq = self.fssm_b[0](x1_b, x1_a)
        Y1_b = y1_b_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        x2_a, x2_b = torch.flip(x1_a, dims=[1]), torch.flip(x1_b, dims=[1])
        y2_b_seq = torch.flip(self.fssm_b[1](x2_b, x2_a), dims=[1])
        Y2_b = y2_b_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        x3_a = Xa.permute(0, 3, 2, 1).contiguous().view(B, W * H, C)
        x3_b = Xb.permute(0, 3, 2, 1).contiguous().view(B, W * H, C)
        y3_b_seq = self.fssm_b[2](x3_b, x3_a)
        Y3_b = y3_b_seq.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        
        x4_a, x4_b = torch.flip(x3_a, dims=[1]), torch.flip(x3_b, dims=[1])
        y4_b_seq = torch.flip(self.fssm_b[3](x4_b, x4_a), dims=[1])
        Y4_b = y4_b_seq.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        
        Yb = Y1_b + Y2_b + Y3_b + Y4_b
        
        # --- branch A (slice) - only when required ---
        Ya = None
        if self.calc_a:
            y1_a_seq = self.fssm_a[0](x1_a, x1_b)
            Y1_a = y1_a_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            
            y2_a_seq = torch.flip(self.fssm_a[1](x2_a, x2_b), dims=[1])
            Y2_a = y2_a_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            
            y3_a_seq = self.fssm_a[2](x3_a, x3_b)
            Y3_a = y3_a_seq.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
            
            y4_a_seq = torch.flip(self.fssm_a[3](x4_a, x4_b), dims=[1])
            Y4_a = y4_a_seq.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
            
            Ya = Y1_a + Y2_a + Y3_a + Y4_a
            
        return Ya, Yb

class SSVF(nn.Module):
    def __init__(self, channels, d_state=32, expand=2, device=None, only_axial=True):
        super().__init__()
        self.only_axial = only_axial
        self.C = channels
        self.d_inner = int(expand * channels)

        # 2D: projection only, no gate or output (one-way)
        self.norm_S = nn.LayerNorm(channels)
        self.convS_x = nn.Conv2d(channels, self.d_inner, kernel_size=1, bias=False, device=device)

        # 3D: voll (projection + gate + output)
        self.norm_V = nn.LayerNorm(channels)
        self.convV_x = nn.Conv3d(channels, self.d_inner, kernel_size=1, bias=False, device=device)
        self.convV_z = nn.Conv3d(channels, self.d_inner, kernel_size=1, bias=False, device=device)
        self.convV_o = nn.Conv3d(self.d_inner, channels, kernel_size=1, bias=False, device=device)

        # Scans: calc_a=False → kein Slice-Update, spart 50%
        self.scan_axial = _FeatureSSVFScan(C=self.d_inner, d_state=d_state, calc_a=False)

        if not self.only_axial:
            self.scan_coronal = _FeatureSSVFScan(C=self.d_inner, d_state=d_state, calc_a=False)
            self.scan_sagittal = _FeatureSSVFScan(C=self.d_inner, d_state=d_state, calc_a=False)

    def forward(self, slice_2d, vol_3d):
        B, C, D, H, W = vol_3d.shape

        # 1. Norm & Project
        S_norm = self.norm_S(slice_2d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        V_norm = self.norm_V(vol_3d.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()

        Sx = self.convS_x(S_norm)
        Vx = self.convV_x(V_norm)
        Vz = self.convV_z(V_norm)

        # 2. Projective Squeeze
        sx_ax = F.interpolate(Sx, size=(H, W), mode='bilinear', align_corners=False)

        # 3. tri-plane sweep - slice as query, the volume is scanned
        # Axial
        sx_ax = sx_ax.unsqueeze(1).expand(-1, D, -1, -1, -1).reshape(B * D, self.d_inner, H, W)
        vx_ax = Vx.permute(0, 2, 1, 3, 4).reshape(B * D, self.d_inner, H, W)
        _, yv_ax = self.scan_axial(sx_ax, vx_ax)
        YV_ax = yv_ax.view(B, D, self.d_inner, H, W).permute(0, 2, 1, 3, 4)

        if self.only_axial:
            out_vol = self.convV_o(YV_ax * F.silu(Vz)) + vol_3d
            return slice_2d, out_vol

        sx_co = F.interpolate(Sx, size=(D, W), mode='bilinear', align_corners=False)
        sx_sa = F.interpolate(Sx.permute(0, 1, 3, 2).contiguous(), size=(D, H), mode='bilinear', align_corners=False)

        # Coronal
        sx_co = sx_co.unsqueeze(1).expand(-1, H, -1, -1, -1).reshape(B * H, self.d_inner, D, W)
        vx_co = Vx.permute(0, 3, 1, 2, 4).reshape(B * H, self.d_inner, D, W)
        _, yv_co = self.scan_coronal(sx_co, vx_co)
        YV_co = yv_co.view(B, H, self.d_inner, D, W).permute(0, 2, 3, 1, 4)

        # Sagittal
        sx_sa = sx_sa.unsqueeze(1).expand(-1, W, -1, -1, -1).reshape(B * W, self.d_inner, D, H)
        vx_sa = Vx.permute(0, 4, 1, 2, 3).reshape(B * W, self.d_inner, D, H)
        _, yv_sa = self.scan_sagittal(sx_sa, vx_sa)
        YV_sa = yv_sa.view(B, W, self.d_inner, D, H).permute(0, 2, 3, 4, 1)

        # 4. Reassembly + Gating + Residual
        YV_total = YV_ax + YV_co + YV_sa
        out_vol = self.convV_o(YV_total * F.silu(Vz)) + vol_3d

        return slice_2d, out_vol


# ==========================================================================
# CWCD  — correlation-weighted coordinate decoder (SSVF + depth reduction)
# ==========================================================================


class DepthReduction(nn.Module):
    """
    Reduces the stage-1 output (P coordinate-sum channels + 1 normalizer)
    along ONE axis with per-position positive weights derived from a feature
    signal:  w_d = softplus(c)^T · Decay · softplus(B(f_d)) · softplus(dt(f_d)).

    All P+1 channels share the same weights (shared A row, shared scalar dt),
    so numerators and denominator stay consistent and the caller's single
    division yields a proper weighted expectation. Bidirectional scans are
    SUMMED, not averaged (equivalent after division, and keeps num/den raw).

    forward(x_seq, feat_seq):
        x_seq    : (b, P+1, L)   stage-1 num/den along the axis
        feat_seq : (b, C, L)     gate features along the same axis
        returns  : (b, P+1)      reduced num/den (still undivided)
    """

    def __init__(self, feat_C, N=16, P=3, bidirectional=True):
        super().__init__()
        self.P1 = P + 1
        self.bi = bidirectional
        self.lin_B = nn.Linear(feat_C, N, bias=False)
        self.lin_dt = nn.Linear(feat_C, 1, bias=True)
        self._init_delta_bias(self.lin_dt)
        self.c_read = nn.Parameter(torch.zeros(N))    # constant positive readout
        self.a_log = nn.Parameter(torch.zeros(N))     # A = -exp(a), one shared row

    @staticmethod
    def _init_delta_bias(layer, dt_min=0.001, dt_max=0.1, dt_floor=1e-4):
        c = layer.bias.shape[0]
        dt = torch.exp(torch.rand(c) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            layer.bias.copy_(inv_dt)

    def _scan(self, x_seq, feat_seq):
        b, _, L = x_seq.shape
        f = feat_seq.transpose(1, 2)                                          # (b, L, C)
        Bm = F.softplus(self.lin_B(f)).transpose(1, 2).contiguous()           # (b, N, L)
        Cm = F.softplus(self.c_read).view(1, -1, 1).expand(b, -1, L).contiguous()
        dt = self.lin_dt(f).transpose(1, 2).expand(-1, self.P1, -1).contiguous()
        A = (-torch.exp(self.a_log.float())).unsqueeze(0).expand(self.P1, -1).contiguous()
        y = selective_scan(x_seq, dt, A, Bm, Cm, None)
        return y[:, :, -1]                                                    # (b, P+1)

    def forward(self, x_seq, feat_seq):
        out = self._scan(x_seq, feat_seq)
        if self.bi:
            out = out + self._scan(torch.flip(x_seq, [-1]), torch.flip(feat_seq, [-1]))
        return out

def _plane_coord_grid(plane, B, D, H, W, device, dtype=torch.float32):
    zs = torch.arange(D, device=device, dtype=dtype) - (D - 1) / 2
    ys = torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2
    xs = torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing='ij')
    G = torch.stack([gz, gy, gx], 0).unsqueeze(0).expand(B, -1, -1, -1, -1)
    if plane == 'axial':
        return G.permute(0, 2, 1, 3, 4).reshape(B * D, 3, H, W)
    if plane == 'coronal':
        return G.permute(0, 3, 1, 2, 4).reshape(B * H, 3, D, W)
    if plane == 'sagittal':
        return G.permute(0, 4, 1, 2, 3).reshape(B * W, 3, D, H)
    raise ValueError(plane)

def _slice_identity_grid(Hs, Ws, device, dtype=torch.float32):
    ys = torch.arange(Hs, device=device, dtype=dtype) - (Hs - 1) / 2
    xs = torch.arange(Ws, device=device, dtype=dtype) - (Ws - 1) / 2
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    gz = torch.zeros_like(gy)
    return torch.stack([gz, gy, gx], 0)

class CWCD(nn.Module):
    def __init__(self, channels, d_state=32, expand=2, device=None,
                 only_axial=True, coord_P=3, reduce_N=None, reduce_bi=True,
                 emit_volume=True, emit_conf=False, eps=1e-6):
        super().__init__()
        self.only_axial = only_axial
        self.emit_volume = emit_volume
        self.emit_conf = emit_conf
        self.eps = eps
        self.coord_P = coord_P
        self.C = channels
        self.d_inner = int(expand * channels)
        reduce_N = reduce_N or d_state

        self.norm_S = nn.LayerNorm(channels)
        self.convS_x = nn.Conv2d(channels, self.d_inner, 1, bias=False, device=device)

        self.norm_V = nn.LayerNorm(channels)
        self.convV_x = nn.Conv3d(channels, self.d_inner, 1, bias=False, device=device)
        if emit_volume:
            self.convV_z = nn.Conv3d(channels, self.d_inner, 1, bias=False, device=device)
            self.convV_o = nn.Conv3d(self.d_inner, channels, 1, bias=False, device=device)

        # axial: features + coordinates (reading grid == slice lattice)
        self.scan_axial = SSVFScan(C=self.d_inner, d_state=d_state,
                                       calc_a=False, coord_P=coord_P)
        self.reduce_axial = DepthReduction(feat_C=self.d_inner, N=reduce_N,
                                          P=coord_P, bidirectional=reduce_bi)
        # coronal/sagittal: FEATURE context only (no coordinate scan)
        if not only_axial:
            self.scan_coronal = SSVFScan(C=self.d_inner, d_state=d_state, calc_a=False)
            self.scan_sagittal = SSVFScan(C=self.d_inner, d_state=d_state, calc_a=False)

    def forward(self, slice_2d, vol_3d):
        B, C, D, H, W = vol_3d.shape
        _, _, Hs, Ws = slice_2d.shape
        dev = vol_3d.device

        S_norm = self.norm_S(slice_2d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        V_norm = self.norm_V(vol_3d.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()
        Sx = self.convS_x(S_norm)
        Vx = self.convV_x(V_norm)

        # ---------- AXIAL: features + coordinate num/den ----------
        sx_ax = F.interpolate(Sx, size=(H, W), mode='bilinear', align_corners=False)
        sx_ax = sx_ax.unsqueeze(1).expand(-1, D, -1, -1, -1).reshape(B * D, self.d_inner, H, W)
        vx_ax = Vx.permute(0, 2, 1, 3, 4).reshape(B * D, self.d_inner, H, W)
        g_ax = _plane_coord_grid('axial', B, D, H, W, dev)
        _, yv_ax, yc_ax = self.scan_axial(sx_ax, vx_ax, g_ax)
        YV_total = yv_ax.view(B, D, self.d_inner, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # ---------- CORONAL / SAGITTAL: feature context only ----------
        if not self.only_axial:
            sx_co = F.interpolate(Sx, size=(D, W), mode='bilinear', align_corners=False)
            sx_co = sx_co.unsqueeze(1).expand(-1, H, -1, -1, -1).reshape(B * H, self.d_inner, D, W)
            vx_co = Vx.permute(0, 3, 1, 2, 4).reshape(B * H, self.d_inner, D, W)
            _, yv_co = self.scan_coronal(sx_co, vx_co)
            YV_total = YV_total + yv_co.view(B, H, self.d_inner, D, W).permute(0, 2, 3, 1, 4)

            sx_sa = F.interpolate(Sx.permute(0, 1, 3, 2).contiguous(), size=(D, H),
                                  mode='bilinear', align_corners=False)
            sx_sa = sx_sa.unsqueeze(1).expand(-1, W, -1, -1, -1).reshape(B * W, self.d_inner, D, H)
            vx_sa = Vx.permute(0, 4, 1, 2, 3).reshape(B * W, self.d_inner, D, H)
            _, yv_sa = self.scan_sagittal(sx_sa, vx_sa)
            YV_total = YV_total + yv_sa.view(B, W, self.d_inner, D, H).permute(0, 2, 3, 4, 1)
            YV_total = YV_total.contiguous()

        # ---------- normalized depth reduce (gates: fused features) ----------
        P1 = self.coord_P + 1
        yc = yc_ax.view(B, D, P1, H, W)
        x_seq = yc.permute(0, 3, 4, 2, 1).reshape(B * H * W, P1, D)
        f_seq = YV_total.permute(0, 3, 4, 1, 2).reshape(B * H * W, self.d_inner, D)
        agg = self.reduce_axial(x_seq, f_seq).view(B, H, W, P1).permute(0, 3, 1, 2)

        # single division -> convex combination of volume coordinates
        den = agg[:, self.coord_P:]                              # (B, 1, H, W) normalizer / evidence mass
        coord = agg[:, :self.coord_P] / (den + self.eps)
        coord = F.interpolate(coord, size=(Hs, Ws), mode='bilinear', align_corners=False)

        gs = _slice_identity_grid(Hs, Ws, dev).unsqueeze(0)
        phi = coord - gs

        out_vol = None
        if self.emit_volume:
            Vz = self.convV_z(V_norm)
            out_vol = self.convV_o(YV_total * F.silu(Vz)) + vol_3d

        # weighted-Kabsch confidence: the per-pixel evidence mass, the
        # only correspondence-quality signal the block computes (>= 0 by softplus
        # accumulation). Returned only when requested, so existing 3-tuple callers
        # are unaffected.
        if self.emit_conf:
            conf = F.interpolate(den, size=(Hs, Ws), mode='bilinear', align_corners=False)
            return slice_2d, out_vol, phi, conf
        return slice_2d, out_vol, phi


# ==========================================================================
# WKPS  — weighted Kabsch pose solver (parameter-free)
# ==========================================================================


def _slice_grid(Hs, Ws, device, dtype=torch.float32):
    ys = torch.arange(Hs, device=device, dtype=dtype) - (Hs - 1) / 2
    xs = torch.arange(Ws, device=device, dtype=dtype) - (Ws - 1) / 2
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    gz = torch.zeros_like(gy)
    return torch.stack([gz, gy, gx], 0)

def kps(phi, trans_scale=1.0):
    """
    phi : (B, 3, Hs, Ws) coarse flow (target = source_grid + phi).
    Returns 9D pose [tx,ty,tz, 6d-rot] via the rigid transform that best maps
    the slice grid to (grid + phi). Fully differentiable (SVD).
 
    CONVENTIONS TO VERIFY:
      * order (z,y,x) must match the GT 9D rotation convention.
      * t is in the SAME units as phi (feature-grid). Set trans_scale = output
        stride so t matches GT translation (volume voxels).
      * R here maps source->target (R * Gs + t = Gc). If the GT uses the inverse
        convention, transpose R before extracting the 6D columns.
    """
    B = phi.shape[0]
    _, _, Hs, Ws = phi.shape
    dev = phi.device
 
    gs = _slice_grid(Hs, Ws, dev)                          # (3, Hs, Ws)
    P = gs.reshape(3, -1).t().unsqueeze(0).expand(B, -1, -1)         # (B, N, 3) source
    Q = P + phi.reshape(B, 3, -1).transpose(1, 2)                    # (B, N, 3) target
 
    Pm = P.mean(1, keepdim=True)
    Qm = Q.mean(1, keepdim=True)
    Pc = P - Pm
    Qc = Q - Qm
    Hmat = torch.bmm(Pc.transpose(1, 2), Qc)                         # (B, 3, 3)
 
    U, S, Vh = torch.linalg.svd(Hmat)
    V = Vh.transpose(1, 2)
    Ut = U.transpose(1, 2)
    d = torch.det(torch.bmm(V, Ut))                                 # (B,)
    Dfix = torch.eye(3, device=dev).unsqueeze(0).repeat(B, 1, 1)
    Dfix[:, 2, 2] = d
    R = torch.bmm(torch.bmm(V, Dfix), Ut)                          # (B, 3, 3)
 
    t = (Qm.squeeze(1) - torch.bmm(R, Pm.transpose(1, 2)).squeeze(2)) * trans_scale
    r6 = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)                # (B, 6)
    return torch.cat([t, r6], dim=1)

def wkps(phi, weights, trans_scale=1.0, eps=1e-6):
    """
    Weighted Kabsch / Procrustes fit. Identical to kps, except each
    correspondence i carries a non-negative confidence w_i, so the model can
    down-weight unreliable flow vectors instead of letting them corrupt the
    rigid fit.  Reduces EXACTLY to kps when all weights are equal.

    phi     : (B, 3, Hs, Ws) coarse flow (target = source_grid + phi).
    weights : (B, 1, Hs, Ws) or (B, Hs, Ws), NON-negative. Internally floored
              (+eps) and normalized to sum 1 over the N correspondences, so a
              global scale on `weights` has no effect on the result.
    Conventions (order z,y,x; t in phi units * trans_scale) are as in
    kps. Fully differentiable (SVD).
    """
    B = phi.shape[0]
    _, _, Hs, Ws = phi.shape
    dev = phi.device

    gs = _slice_grid(Hs, Ws, dev)                                   # (3, Hs, Ws)
    P = gs.reshape(3, -1).t().unsqueeze(0).expand(B, -1, -1)        # (B, N, 3) source
    Q = P + phi.reshape(B, 3, -1).transpose(1, 2)                   # (B, N, 3) target

    w = weights.reshape(B, -1, 1).clamp_min(0) + eps               # (B, N, 1) >= eps
    w = w / w.sum(1, keepdim=True)                                  # normalize -> sum 1

    Pm = (w * P).sum(1, keepdim=True)                              # (B, 1, 3) weighted centroid
    Qm = (w * Q).sum(1, keepdim=True)
    Pc = P - Pm
    Qc = Q - Qm
    Hmat = torch.bmm((Pc * w).transpose(1, 2), Qc)                 # (B, 3, 3) weighted covariance

    U, S, Vh = torch.linalg.svd(Hmat)
    V = Vh.transpose(1, 2)
    Ut = U.transpose(1, 2)
    d = torch.det(torch.bmm(V, Ut))                                # (B,)
    Dfix = torch.eye(3, device=dev).unsqueeze(0).repeat(B, 1, 1)
    Dfix[:, 2, 2] = d
    R = torch.bmm(torch.bmm(V, Dfix), Ut)                         # (B, 3, 3)

    t = (Qm.squeeze(1) - torch.bmm(R, Pm.transpose(1, 2)).squeeze(2)) * trans_scale
    r6 = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)               # (B, 6)
    return torch.cat([t, r6], dim=1)


# ==========================================================================
# Ablation: cross-attention substitute for SSVF
# ==========================================================================


def _sincos_1d(length, dim, device, dtype):
    """(length, dim) sin-cos. dim must be even."""
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(dim // 2, device=device, dtype=torch.float32)[None, :]
    freq = torch.exp(-math.log(10000.0) * (2 * i) / dim)
    ang = pos * freq
    return torch.cat([ang.sin(), ang.cos()], dim=1).to(dtype)

def sincos_3d(D, H, W, dim, device, dtype):
    """(D*H*W, dim). dim is split across the three axes (remainder -> W)."""
    d_each = (dim // 6) * 2                      # gerade, pro Achse
    d_w = dim - 2 * d_each
    pd = _sincos_1d(D, d_each, device, dtype)                     # (D, d_each)
    ph = _sincos_1d(H, d_each, device, dtype)                     # (H, d_each)
    pw = _sincos_1d(W, d_w, device, dtype)                        # (W, d_w)
    pe = torch.cat([
        pd[:, None, None, :].expand(D, H, W, d_each),
        ph[None, :, None, :].expand(D, H, W, d_each),
        pw[None, None, :, :].expand(D, H, W, d_w),
    ], dim=-1)
    return pe.reshape(D * H * W, dim)

def sincos_2d(H, W, dim, device, dtype):
    """(H*W, dim)."""
    d_h = (dim // 4) * 2
    d_w = dim - d_h
    ph = _sincos_1d(H, d_h, device, dtype)
    pw = _sincos_1d(W, d_w, device, dtype)
    pe = torch.cat([
        ph[:, None, :].expand(H, W, d_h),
        pw[None, :, :].expand(H, W, d_w),
    ], dim=-1)
    return pe.reshape(H * W, dim)

class CrossAttentionFusion(nn.Module):
    """Cross-Attention-Fusion mit EXAKT der Signatur von VolTriPlaneFusionBlock.

    Structurally identical; only the operator is swapped:

        VolTriPlaneFusionBlock          CrossAttentionFusion
        ----------------------          --------------------
        norm_S, norm_V                  norm_S, norm_V          (same)
        convS_x                         convS_x                 (same)
        convV_x, convV_z                convV_x, convV_z        (same)
        3x FusionScan2D  <-- SWAP -->   Multi-Head Attention
        Y * SiLU(Vz)                    Y * SiLU(Vz)            (same)
        convV_o(...) + vol_3d           convV_o(...) + vol_3d   (same)

    Args:
        channels : encoder channels (enc_ch).
        num_heads: attention heads. d_inner must be divisible by num_heads.
        expand   : d_inner = expand * channels (wie im Mamba-Block).
        include_volume_keys:
            True  -> K/V = [Volumen ++ Slice].  Das faithful Analog des Scans
                     (Volumen-Self-Context, slice-konditioniert). O(N_v(N_v+N_s)).
            False -> K/V = slice only. Pure cross-attention, O(N_s N_v).
                     A second ablation: does the fusion need volume self-context?
        pos      : 'sincos' (shape-agnostisch, empfohlen) | 'none'
        chunk_q  : process queries in blocks to save memory; None = all at once.
    """

    def __init__(self, channels, num_heads=8, expand=2,
                 include_volume_keys=True, pos='sincos',
                 dropout=0.0, chunk_q=None, device=None):
        super().__init__()
        self.C = channels
        self.d_inner = int(expand * channels)
        if self.d_inner % num_heads != 0:
            raise ValueError(f"d_inner={self.d_inner} not divisible by "
                             f"num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = self.d_inner // num_heads
        self.include_volume_keys = include_volume_keys
        self.pos = pos
        self.dropout = dropout
        self.chunk_q = chunk_q

        # --- identisch zu VolTriPlaneFusionBlock ---
        self.norm_S = nn.LayerNorm(channels)
        self.convS_x = nn.Conv2d(channels, self.d_inner, 1, bias=False, device=device)

        self.norm_V = nn.LayerNorm(channels)
        self.convV_x = nn.Conv3d(channels, self.d_inner, 1, bias=False, device=device)
        self.convV_z = nn.Conv3d(channels, self.d_inner, 1, bias=False, device=device)
        self.convV_o = nn.Conv3d(self.d_inner, channels, 1, bias=False, device=device)

        # --- der getauschte Operator ---
        self.to_q = nn.Linear(self.d_inner, self.d_inner, bias=False, device=device)
        self.to_k = nn.Linear(self.d_inner, self.d_inner, bias=False, device=device)
        self.to_v = nn.Linear(self.d_inner, self.d_inner, bias=False, device=device)
        self.proj = nn.Linear(self.d_inner, self.d_inner, bias=False, device=device)

    # ---- helpers ----
    def _heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _unheads(self, x):
        B, h, N, hd = x.shape
        return x.transpose(1, 2).reshape(B, N, h * hd)

    def _attend(self, q, k, v):
        """SDPA -> FlashAttention: O(N) memory, no score matrix."""
        if self.chunk_q is None:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0)
        outs = []
        for i in range(0, q.shape[2], self.chunk_q):
            outs.append(F.scaled_dot_product_attention(
                q[:, :, i:i + self.chunk_q], k, v,
                dropout_p=self.dropout if self.training else 0.0))
        return torch.cat(outs, dim=2)

    # ---- forward: gleiche Signatur wie VolTriPlaneFusionBlock ----
    def forward(self, slice_2d, vol_3d):
        B, C, D, H, W = vol_3d.shape
        dev, dt = vol_3d.device, vol_3d.dtype

        # 1. Norm & Project  (identisch zum Mamba-Block)
        S_norm = self.norm_S(slice_2d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        V_norm = self.norm_V(vol_3d.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()

        Sx = self.convS_x(S_norm)                      # (B, d_inner, Hs, Ws)
        Vx, Vz = self.convV_x(V_norm), self.convV_z(V_norm)

        # 2. Tokens
        vol_tok = Vx.flatten(2).transpose(1, 2)        # (B, N_v, d_inner)
        sli_tok = Sx.flatten(2).transpose(1, 2)        # (B, N_s, d_inner)

        # 3. positional encoding - attention needs it, the scan does not
        if self.pos == 'sincos':
            vol_tok = vol_tok + sincos_3d(D, H, W, self.d_inner, dev, dt)[None]
            Hs, Ws = Sx.shape[-2:]
            sli_tok = sli_tok + sincos_2d(Hs, Ws, self.d_inner, dev, dt)[None]

        # 4. Attention — DER getauschte Operator
        kv = torch.cat([vol_tok, sli_tok], dim=1) if self.include_volume_keys else sli_tok
        q = self._heads(self.to_q(vol_tok))
        k = self._heads(self.to_k(kv))
        v = self._heads(self.to_v(kv))
        y = self.proj(self._unheads(self._attend(q, k, v)))     # (B, N_v, d_inner)

        # 5. Reassembly + Gating + Residual  (identisch zum Mamba-Block)
        Y = y.transpose(1, 2).view(B, self.d_inner, D, H, W)
        out_vol = self.convV_o(Y * F.silu(Vz)) + vol_3d

        return slice_2d, out_vol


# ==========================================================================
# Ablation: direct pose regression heads
# ==========================================================================


class TranslationHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        return self.head(x)

class RotationHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 6),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, x):
        return self.head(x)

class PhiEncoder(nn.Module):
    def __init__(self, enc_ch, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, enc_ch, 3, stride=2, padding=1), nn.GroupNorm(8, enc_ch), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
 
    def forward(self, phi):
        return self.net(phi).flatten(1)


# ==========================================================================
# SCoPE-Reg
# ==========================================================================


class _ConfHead(nn.Module):
    """Per-pixel confidence logits from the coarse flow phi (B,3,Hs,Ws)->(B,1,Hs,Ws).

    Used by kabsch_weight='learned'. Zero-init'd last conv so weights start
    ~uniform (softplus(0)), i.e. the fit begins as the unweighted Kabsch.
    """

    def __init__(self, in_ch=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phi):
        return self.net(phi)

class SCoPEReg(nn.Module):
    """The reported model: CWCD for correspondences, WKPS for the pose.

    The encoder is fixed (ResNet-8, strides (1,2,1) -> total stride 8), so the
    correspondence grid is always input/8 and the parameter count does not
    depend on the input resolution.
    """

    ENCODER_STRIDE = (1, 2, 1)

    def __init__(self, d_state=32, expand=2, fusion_kwargs=None,
                 phi_trans_scale=8.0, kabsch_weight='fusion_conf'):
        super().__init__()
        if kabsch_weight not in ('uniform', 'fusion_conf', 'learned'):
            raise ValueError("kabsch_weight must be uniform|fusion_conf|learned, "
                             f"got '{kabsch_weight}'")
        self.out_dof = 9
        self.phi_trans_scale = phi_trans_scale
        self.kabsch_weight = kabsch_weight

        fusion_kwargs = dict(fusion_kwargs or {})
        fusion_kwargs['emit_volume'] = False      # WKPS needs no fused volume
        fusion_kwargs['emit_conf'] = (kabsch_weight == 'fusion_conf')

        self.encoder_mr = ResNetEncoder8_2D(in_channels=1,
                                            layer_strides=self.ENCODER_STRIDE)
        self.encoder_ct = ResNetEncoder8_3D(in_channels=1,
                                            layer_strides=self.ENCODER_STRIDE)
        self.enc_channels = self.encoder_mr.out_channels

        self.fusion_block = CWCD(channels=self.enc_channels, d_state=d_state,
                                 expand=expand, **fusion_kwargs)
        self.conf_head = _ConfHead(in_ch=3) if kabsch_weight == 'learned' else None

    def forward(self, mr_slice, ct_vol):
        slice_ctx = self.encoder_mr(mr_slice.float())
        vol_ctx = self.encoder_ct(ct_vol.float())
        out = self.fusion_block(slice_ctx, vol_ctx)

        if self.kabsch_weight == 'fusion_conf':
            _, _, phi, conf = out                              # block emitted evidence mass
            pred = wkps(phi, conf, trans_scale=self.phi_trans_scale)
        elif self.kabsch_weight == 'learned':
            _, _, phi = out
            w = F.softplus(self.conf_head(phi))                # (B,1,Hs,Ws) >= 0
            pred = wkps(phi, w, trans_scale=self.phi_trans_scale)
        else:                                                  # 'uniform' -> unweighted Kabsch
            _, _, phi = out
            pred = kps(phi, trans_scale=self.phi_trans_scale)
        return pred, phi

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {'model': total}, total


# ==========================================================================
# Ablation model: the same encoders and fusion, but a regression head instead
# of the solver. Covers the three "Direct Heads" rows of the ablation table.
# ==========================================================================

class HeadsRegModel(nn.Module):
    """Direct pose regression: pooled descriptor -> two MLP heads.

    `fusion` selects what produces the descriptor:
      'cross_attn'     CrossAttentionFusion -> pooled fused volume
      'triplane_fssm'  SSVF                 -> pooled fused volume
      'triplane_coord' CWCD                 -> coordinate field -> PhiEncoder
    """

    ENCODER_STRIDE = (1, 2, 1)

    def __init__(self, fusion='triplane_fssm', d_state=32, expand=2,
                 fusion_kwargs=None):
        super().__init__()
        self.out_dof = 9
        self.fusion_name = fusion
        fusion_kwargs = dict(fusion_kwargs or {})

        self.encoder_mr = ResNetEncoder8_2D(in_channels=1,
                                            layer_strides=self.ENCODER_STRIDE)
        self.encoder_ct = ResNetEncoder8_3D(in_channels=1,
                                            layer_strides=self.ENCODER_STRIDE)
        C = self.encoder_mr.out_channels
        self.enc_channels = C

        if fusion == 'cross_attn':
            fusion_kwargs.pop('only_axial', None)
            self.fusion_block = CrossAttentionFusion(C, expand=expand, **fusion_kwargs)
            self.from_coord = False
        elif fusion == 'triplane_fssm':
            self.fusion_block = SSVF(C, d_state=d_state, expand=expand, **fusion_kwargs)
            self.from_coord = False
        elif fusion == 'triplane_coord':
            fusion_kwargs['emit_volume'] = False    # pose comes from the field
            self.fusion_block = CWCD(channels=C, d_state=d_state, expand=expand,
                                     **fusion_kwargs)
            self.from_coord = True
        else:
            raise ValueError(f"unknown fusion {fusion!r}")

        self.phi_encoder = PhiEncoder(C) if self.from_coord else None
        self.pool = None if self.from_coord else nn.AdaptiveAvgPool3d(1)
        self.trans_head = TranslationHead(C)
        self.rot_head = RotationHead(C)

    def forward(self, mr_slice, ct_vol):
        slice_ctx = self.encoder_mr(mr_slice.float())
        vol_ctx = self.encoder_ct(ct_vol.float())
        out = self.fusion_block(slice_ctx, vol_ctx)

        if self.from_coord:
            phi = out[2] if len(out) >= 3 else out[1]
            feat = self.phi_encoder(phi)
            pred = torch.cat([self.trans_head(feat), self.rot_head(feat)], dim=1)
            return pred, phi

        vol_fused = out[1] if isinstance(out, (tuple, list)) else out
        feat = self.pool(vol_fused).flatten(1)
        return torch.cat([self.trans_head(feat), self.rot_head(feat)], dim=1)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {'model': total}, total


# ==========================================================================

ENCODER_TOTAL_STRIDE = 8        # ResNet-8 stem (4) x stage strides (1,2,1)


def build_model(cfg, device, volume_shape=None):
    """Build the model described by cfg['model']."""
    v = cfg["model"]
    fusion_kwargs = dict(v.get("fusion_kwargs", {}))
    common = dict(d_state=v["d_state"], expand=v.get("expand", 2),
                  fusion_kwargs=fusion_kwargs)

    if v["variant"] == "procrustes":
        if v["fusion"] != "triplane_coord":
            raise ValueError("the WKPS readout requires fusion='triplane_coord'")
        if v.get("reduce_n") is not None:
            fusion_kwargs.setdefault("reduce_N", v["reduce_n"])
        fusion_kwargs.setdefault("reduce_bi", v.get("reduce_bidirectional", True))
        model = SCoPEReg(phi_trans_scale=float(ENCODER_TOTAL_STRIDE),
                         kabsch_weight=v.get("kabsch_weight", "fusion_conf"),
                         **common)
        model.svr_name = "SCoPE-Reg (CWCD + WKPS)"
    else:
        if v["fusion"] == "triplane_coord":
            if v.get("reduce_n") is not None:
                fusion_kwargs.setdefault("reduce_N", v["reduce_n"])
            fusion_kwargs.setdefault("reduce_bi", v.get("reduce_bidirectional", True))
        model = HeadsRegModel(fusion=v["fusion"], **common)
        model.svr_name = f"{v['fusion']} + Direct Heads"

    model = model.to(device)
    model.out_dof = 9
    return model
