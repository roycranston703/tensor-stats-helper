# tensor-stats-helper
Lightweight utilities for quick tensor statistics during prototyping.

## Installation
```bash
pip install tensor-stats-helper
``` 

<details>
<summary>🧠 Additional examples (click to expand)</summary>

<pre><code class="language-python">
#  SECTION 0 + 1  ·  GLOBAL CONFIG  +  DATA I/O  +  SANITY

# Purpose
#   • Central CFG dict: tweak here → propagates everywhere.
#   • Shape-agnostic loader: survives extra/missing channels,
#     higher resolutions, or time-stacked inputs.
#   • Quick sanity asserts to catch label flips & data corruption.

# ---------- imports ----------
from pathlib import Path
import random, torch, torch.nn as nn, torch.utils.data as td

# ---------- editable global hyper-parameters ----------
CFG = {
    # will auto-update ‘in_ch’ and ‘img_shape’ after first batch
    "in_ch":      6,            # expected channels; used for asserts
    "out_ch":     2,            # change to N for multi-class
    "pos_weight": 1500.0,       # background : foreground weight
    "batch":      64,
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}

# ---------- core loader ----------
def load_sample(pt_path: Path):
    """Return (x, y) tensors from a saved .pt file.
       - x: C×H×W  float32
       - y:     H×W  int64   (class indices)"""
    t = torch.load(pt_path)          # raw tensor saved by organisers
    x, y = t[:-1].float(), t[-1].long()
    return x, y

def make_loader(data_dir, split="train", batch=CFG["batch"]):
    """Shape-agnostic DataLoader ready for any #channels / resolution."""
    files = sorted(Path(data_dir).glob("*.pt"))
    if split == "train":
        random.shuffle(files)

    xs, ys = zip(*(load_sample(f) for f in files))
    xs, ys = torch.stack(xs), torch.stack(ys)        # B×C×H×W ,  B×H×W

    # ---- auto-update CFG on first call ----
    if CFG["in_ch"] != xs.size(1):           # extra / missing channels
        CFG["in_ch"] = xs.size(1)
    if "img_shape" not in CFG or CFG["img_shape"] != xs.shape[-2:]:
        CFG["img_shape"] = xs.shape[-2:]     # e.g. (50,181) → (100,361)

    ds = td.TensorDataset(xs, ys)
    return td.DataLoader(ds, batch_size=batch, shuffle=(split == "train"))

# ---------- fast sanity check ----------
def sanity_batch(x, y):
    """Abort early if something is obviously wrong."""
    B, C, H, W = x.shape
    assert y.shape == (B, H, W),   "Label dims mismatch ⟹ check loader."
    assert C == CFG["in_ch"],      "Unexpected channel count."
    # Positive-pixel ratio flag for label corruption
    pos_ratio = y.float().mean().item()
    assert pos_ratio < 0.02,       f"Suspiciously high positives ({pos_ratio:.3%})."

# Example usage
if __name__ == "__main__":
    dl = make_loader("/path/to/train", "train")
    x, y = next(iter(dl))
    sanity_batch(x, y)
    print("Loaded batch OK:", x.shape, y.shape, CFG)

# Section 2 · Augmentation Arsenal
# Drag-drop functions; toggle inside apply_train_aug().
# Comments explain what / why each aug helps for possible task extensions.

import torch, math, torch.nn.functional as F

# horizontal flip along azimuth axis – safe if dataset is ±π-symmetric
def random_azimuth_flip(x, y):
    if torch.rand(1) < 0.5:
        x = torch.flip(x, [-1]); y = torch.flip(y, [-1])
    return x, y

# roll ±1 pixel in range – mimics slight distance shift, combats over-fit to exact bins
def range_jitter(x, y, max_px: int = 1):
    shift = int(torch.randint(-max_px, max_px + 1, (1,)))
    if shift:
        x = torch.roll(x, shift, dims=-2)
        y = torch.roll(y, shift, dims=-2)
    return x, y

# randomly zero entire channels – trains resilience to missing sensor slices
def channel_drop(x, y, p: float = 0.15):
    mask = torch.rand(x.size(1), device=x.device) > p
    return x * mask[:, None, None], y

# additive Gaussian noise – guards unseen SNR or weather conditions
def add_noise(x, y, sigma: float = 0.02):
    return x + sigma * torch.randn_like(x), y

# MixUp at batch level – reduces label noise sensitivity; keep commented unless BCE/Dice loss handles soft labels
def mixup_batch(x, y, alpha: float = 0.4):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    # for hard-label segmentation you may choose y or y[idx] based on lam; here we keep dominant
    return x_mix, y if lam >= 0.5 else y[idx]

# convert polar grid to Cartesian coords and concatenate as two extra channels – useful if organisers ask for Cartesian output
def polar_to_cartesian(x):
    B, C, H, W = x.shape
    rng = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
    az  = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
    x_cart = rng * torch.cos(az * math.pi)
    y_cart = rng * torch.sin(az * math.pi)
    return torch.cat([x, x_cart, y_cart], 1)

# master switchboard – edit order / comment lines to suit
def apply_train_aug(x, y):
    x, y = random_azimuth_flip(x, y)
    x, y = range_jitter(x, y)
    x, y = channel_drop(x, y)
    x, y = add_noise(x, y)
    # x, y = mixup_batch(x, y)      # enable if using soft-label-friendly loss
    return x, y

# Section 3 · Mutation-to-Hook Map
# One-glance guide: if the organisers change ⟨X⟩, jump to the code hook noted.

# Input tensor  ────────────────────────────────────────────────────────────
#   examples  : extra / missing channels, higher resolution, time stacks
#   hooks     : Section 4  build_first_conv, Conv3dStem
#               Section 2  channel_drop (training aug)

# Metadata conditioning  ──────────────────────────────────────────────────
#   examples  : scalar (temperature), vector (GPS pose)
#   hooks     : Section 4  FiLM block

# Label space  ────────────────────────────────────────────────────────────
#   examples  : multi-class (human / cyclist / vehicle),
#               multi-label, per-instance masks
#   hooks     : Section 5  MultiClassHead, MultiLabelHead, InstanceCenterHead

# Output geometry  ────────────────────────────────────────────────────────
#   examples  : Cartesian map instead of polar, dual-task polar+cartesian
#   hooks     : Section 2  polar_to_cartesian
#               Section 5  attach_head with separate outputs

# Scoring metric  ─────────────────────────────────────────────────────────
#   examples  : mIoU, Dice, F1@0.9, mAP, new weighted accuracy
#   hooks     : Section 6  metric registry + threshold sweeper   (to add)

# Domain shift  ───────────────────────────────────────────────────────────
#   examples  : rain vs sunny, indoor vs outdoor, lower SNR
#   hooks     : Section 2  aug bag (noise, jitter, MixUp)
#               Section 10 TTA & Section 12 robustness tests     (to add)

# Section 4 · Input & Metadata Adaptation Hooks
# Plug-and-play utilities that let the same backbone survive extra channels,
# time-stacked inputs, or side-information (GPS, temperature, etc.).

import torch
import torch.nn as nn

# Flexible 2-D stem – swap this for the fixed first conv in any model
def build_first_conv(in_ch, out_ch=64, k=3, p=1):
    # Use bias=False so weight shape = (out_ch, in_ch, k, k) can be expanded later
    return nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)

# 3-D stem for inputs shaped B×T×C×H×W (time stacks or micro-Doppler slices)
class Conv3dStem(nn.Module):
    # squeeze time with Conv3d → Conv2d so main network remains 2-D
    def __init__(self, in_ch_per_frame, out_ch=64, t_kernel=3):
        super().__init__()
        self.conv3d = nn.Conv3d(1, out_ch, (t_kernel, 3, 3),
                                padding=(t_kernel // 2, 1, 1))
    def forward(self, x):             # x : B×T×C×H×W   (C usually 6)
        B, T, C, H, W = x.shape
        x = x.view(B, 1, T * C, H, W) # merge time & channel dim
        x = self.conv3d(x)            # B×out_ch×H×W after squeeze
        return x.squeeze(2)

# FiLM (Feature-wise Linear Modulation) block for scalar / vector metadata
class FiLM(nn.Module):
    """
    Example: condition the mid-layer of UNet on a 4-D GPS+temperature vector.
    Usage: feats = film(feats, meta)  # feats: B×C×H×W, meta: B×cond_dim
    """
    def __init__(self, feat_ch: int, cond_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_ch * 2)   # scale and shift
        )

    def forward(self, feats, meta):
        gam_beta = self.net(meta)            # B×2C
        gamma, beta = gam_beta.chunk(2, dim=1)
        gamma = gamma.view(-1, feats.size(1), 1, 1)
        beta  = beta.view(-1, feats.size(1), 1, 1)
        return feats * (1 + gamma) + beta

# ------------------------------------------------------------------
# Example integration snippets (copy into your model definition)
#
# 1. Replace the first conv layer:
#    model.stem = build_first_conv(CFG["in_ch"], out_ch=64)
#
# 2. For time stacks (T×C×H×W), preprocess in forward():
#    x = Conv3dStem(CFG["in_ch"])(x)   # then feed into 2-D backbone
#
# 3. Inject FiLM after encoder mid-block:
#    self.film = FiLM(feat_ch=128, cond_dim=meta_dim)
#    ...
#    feats = self.film(feats, meta_vector)
#
# These hooks cover Mutation-Table rows:
#   • Input tensor changes (channels, resolution, time stacks)
#   • Metadata conditioning (scalar or vector side-info)

# Section 5 · Output-Head Variants
# Drop-in blocks that let the same backbone cope with a
#  • multi-class semantic map
#  • multi-label (one-vs-all) targets
#  • per-instance masks via centre-heat-map + offset regression
# Call attach_head(backbone, ...) once after building the UNet.

import torch.nn as nn

# ---------- simple heads ----------

class MultiClassHead(nn.Module):
    # one softmax logit per class (C_out ≥ 2)
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, 1)
    def forward(self, feats):
        return self.conv(feats)         # use CrossEntropy / Dice

class MultiLabelHead(nn.Module):
    # one sigmoid logit per independent label
    def __init__(self, in_ch: int, num_labels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_labels, 1)
    def forward(self, feats):
        return self.conv(feats)         # use BCE / BCE-Dice per channel

# ---------- instance-mask head ----------
# Predicts a 1-channel object-centre heat-map  + 2-channel XY offset field
class InstanceCenterHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.center = nn.Conv2d(in_ch, 1, 1)
        self.offset = nn.Conv2d(in_ch, 2, 1)
    def forward(self, feats):
        return {
            "center": self.center(feats),          # supervise with focal/Dice
            "offset": self.offset(feats),          # L1 / smooth-L1 to gt offsets
        }

# ---------- helper to stick a head onto any encoder-decoder backbone ----------
def attach_head(backbone: nn.Module,
                head_type: str = "multi_class",
                num_classes: int = 2):
    """
    head_type:  'multi_class' | 'multi_label' | 'instance'
    backbone must expose   backbone.out_channels   (feats coming into head).
    """
    in_ch = getattr(backbone, "out_channels", 64)   # fallback if attr missing
    if head_type == "multi_class":
        head = MultiClassHead(in_ch, num_classes)
    elif head_type == "multi_label":
        head = MultiLabelHead(in_ch, num_classes)
    elif head_type == "instance":
        head = InstanceCenterHead(in_ch)
    else:
        raise ValueError(f"Unknown head_type {head_type}")
    backbone.head = head            # simple attribute; call inside forward
    return backbone

# ---------------------- usage notes ----------------------
# • Multi-class: organisers add cyclist / vehicle → set CFG["out_ch"]=N,
#   rebuild UNet, then attach_head(unet,'multi_class',N).
# • Multi-label: predict human *and* ‘moving’ flags simultaneously.
# • Instance: if task switches to counting individuals, attach instance head
#   and train with a combo of centre-heat-map focal loss + offset L1.

# Section 6 · Metric Registry + Threshold Sweeper
# Collect all scoring rules the organisers might switch to.
# Call metric_fn(logits, y) → float  (logits = raw model output, shape B×C×H×W)
# If a metric needs binarisation, register it with needs_thr=True
# then run sweep_threshold() once to store the best cut-off.

import json, numpy as np, torch
import torch.nn.functional as F

# ---------- core metrics ----------

def organiser_weighted_accuracy(logits, y, pos_w=CFG["pos_weight"]):
    preds = logits.argmax(1)
    pos = (y == 1)
    bg_correct = (preds == 0) & (~pos)
    fg_correct = (preds == 1) & pos
    score = bg_correct.sum() + pos_w * fg_correct.sum()
    max_score = (~pos).sum() + pos_w * pos.sum()
    return (score / max_score).item()

def dice_score(logits, y, thr=0.5):
    probs = torch.softmax(logits, 1)[:, 1]
    preds = (probs > thr).float()
    inter = (preds * (y == 1)).sum()
    union = preds.sum() + (y == 1).sum()
    return (2 * inter / (union + 1e-6)).item()

def iou_score(logits, y, thr=0.5):
    probs = torch.softmax(logits, 1)[:, 1]
    preds = (probs > thr).float()
    inter = (preds * (y == 1)).sum()
    union = preds.sum() + (y == 1).sum() - inter
    return (inter / (union + 1e-6)).item()

def f1_at_thr(logits, y, thr=0.9):
    probs = torch.softmax(logits, 1)[:, 1]
    preds = (probs > thr).float()
    tp = (preds * (y == 1)).sum()
    prec = tp / (preds.sum() + 1e-6)
    rec  = tp / ((y == 1).sum() + 1e-6)
    return (2 * prec * rec / (prec + rec + 1e-6)).item()

# ---------- registry ----------
METRICS = {
    "weighted": dict(fn=organiser_weighted_accuracy, needs_thr=False),
    "dice":     dict(fn=dice_score,                needs_thr=True),
    "iou":      dict(fn=iou_score,                 needs_thr=True),
    "f1@0.9":   dict(fn=lambda l,y: f1_at_thr(l,y,0.9), needs_thr=False)
}

# ---------- threshold sweep for metrics that need one ----------
def sweep_threshold(model, loader, metric_name="dice", steps=50):
    assert METRICS[metric_name]["needs_thr"], "Metric doesn’t need threshold."
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            logits_all.append(model(x.to(CFG["device"])).cpu())
            y_all.append(y)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)

    best_thr, best_val = 0.5, -1
    for thr in np.linspace(0.05, 0.95, steps):
        val = METRICS[metric_name]["fn"](logits, y, thr)
        if val > best_val:
            best_val, best_thr = val, thr

    json.dump(dict(metric=metric_name, thr=best_thr),
              open("best_thr.json", "w"))
    print(f"[sweep] {metric_name}: best {best_val:.4f} @ thr={best_thr:.2f}")

# ---------- inference helper ----------
def apply_metric(model, loader, metric_name="weighted"):
    meta = METRICS[metric_name]
    thr = None
    if meta["needs_thr"]:
        thr = json.load(open("best_thr.json"))["thr"]
    model.eval(); vals = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(CFG["device"]))
            vals.append(meta["fn"](logits.cpu(), y, thr) if thr else meta["fn"](logits.cpu(), y))
    return sum(vals) / len(vals)

# Section 7 · Best-at-Home Res-UNet (CBAM attention optional)

import torch, torch.nn as nn

# --- optional attention ---
class CBAM(nn.Module):
    def __init__(self, ch, red=16, k=7):
        super().__init__()
        self.mlp  = nn.Sequential(nn.Linear(ch, ch // red), nn.ReLU(),
                                  nn.Linear(ch // red, ch))
        self.conv = nn.Conv2d(2, 1, k, padding=(k - 1) // 2)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.shape
        att = self.mlp(x.mean((2, 3)).view(b, c)) + \
              self.mlp(x.amax((2, 3)).view(b, c))
        x   = x * self.sig(att).view(b, c, 1, 1)
        spa = self.sig(self.conv(torch.cat([x.mean(1, True),
                                            x.amax(1, True)], 1)))
        return x * spa

class Identity(nn.Module):          # used when attention is off
    def forward(self, x): return x

# --- building blocks ---
class ResBlock(nn.Module):
    def __init__(self, ci, co, use_attn=True):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, ci)
        self.conv1 = nn.Conv2d(ci, co, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, co)
        self.conv2 = nn.Conv2d(co, co, 3, 1, 1)
        self.act   = nn.SiLU()
        self.skip  = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        self.attn  = CBAM(co) if use_attn else Identity()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.act(self.norm2(self.conv1(h)))
        h = self.conv2(h) + self.skip(x)
        return self.attn(h)

class Down(nn.Module):
    def __init__(self, ci, co, use_attn=True):
        super().__init__()
        self.res1, self.res2 = ResBlock(ci, co, use_attn), ResBlock(co, co, use_attn)
        self.down = nn.Conv2d(co, co, 3, 2, 1)
    def forward(self, x):
        h = self.res1(x); h = self.res2(h)
        return h, self.down(h)

class Up(nn.Module):
    def __init__(self, ci, co, use_attn=True):
        super().__init__()
        self.up   = nn.ConvTranspose2d(ci, ci, 4, 2, 1)
        self.conv = nn.Conv2d(ci + co, co, 3, 1, 1)
        self.res  = ResBlock(co, co, use_attn)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], 1)
        return self.res(self.conv(x))

# --- main UNet ---
class ResUNetCBAM(nn.Module):
    """
    Arguments
        in_ch     : number of input channels  (will set from CFG["in_ch"])
        out_ch    : number of output classes
        use_attn  : True = CBAM, False = disable attention
    Replace first conv if channels differ:
        model.stem = build_first_conv(CFG["in_ch"], 64)    # from Section 4
    Attach multi-class / instance heads via attach_head()  # Section 5
    """
    def __init__(self, in_ch=6, out_ch=2, chs=(64, 64, 128), use_attn=True):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, chs[0], 3, 1, 1)
        self.enc1 = Down(chs[0], chs[0], use_attn)
        self.enc2 = Down(chs[0], chs[1], use_attn)
        self.mid  = ResBlock(chs[1], chs[2], use_attn)
        self.up1  = Up(chs[2], chs[1], use_attn)
        self.up2  = Up(chs[1], chs[0], use_attn)
        self.out_channels = chs[0]          # for Section 5 head attach
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x, meta=None):
        s0       = self.stem(x)
        h1, x1   = self.enc1(s0)
        h2, x2   = self.enc2(x1)
        m        = self.mid(x2)
        u1       = self.up1(m, h2)
        u2       = self.up2(u1, h1)
        return self.head(u2)

# --- build example ---
# model = ResUNetCBAM(in_ch=CFG["in_ch"], out_ch=CFG["out_ch"], use_attn=False)

# Section 8 · Loss Bank
# Pick one that matches the organiser’s scoring rule & class imbalance.
# Instantiate with   criterion = LOSS_BANK["combo"](alpha=0.5)

import torch, torch.nn as nn, torch.nn.functional as F

# --- helpers ---
def _one_hot(y, num_classes=2):
    return F.one_hot(y, num_classes).permute(0, 3, 1, 2).float()

# --- core losses ---
def weighted_bce():
    pos_w = torch.tensor([CFG["pos_weight"]], device=CFG["device"])
    # BCEWithLogits operates on logits[:,1] vs y.float()
    return nn.BCEWithLogitsLoss(pos_weight=pos_w)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1): super().__init__(); self.smooth = smooth
    def forward(self, logits, y):
        probs = torch.softmax(logits, 1)[:, 1]            # foreground prob
        y_f   = y.float()
        inter = (probs * y_f).sum()
        union = probs.sum() + y_f.sum()
        return 1 - (2 * inter + self.smooth) / (union + self.smooth)

class ComboLoss(nn.Module):
    # α weight on BCE, (1-α) on Dice – recommended default 0.5
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce   = weighted_bce()
        self.dice  = DiceLoss()
    def forward(self, logits, y):
        bce  = self.bce(logits[:, 1], y.float())          # BCE on fg logit
        dice = self.dice(logits, y)
        return self.alpha * bce + (1 - self.alpha) * dice

class FocalTverskyLoss(nn.Module):
    # Good when false positives punish the metric heavily
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1):
        super().__init__()
        self.a, self.b, self.g, self.s = alpha, beta, gamma, smooth
    def forward(self, logits, y):
        p = torch.softmax(logits, 1)[:, 1]
        y_f = y.float()
        tp = (p * y_f).sum(); fp = ((1 - y_f) * p).sum(); fn = (y_f * (1 - p)).sum()
        tv = (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)
        return (1 - tv) ** self.g

# Lovász hinge – direct mIoU surrogate (binary)
def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inters = gts - gt_sorted.cumsum(0)
    unions = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - inters / unions
    if gt_sorted.numel() == 0: return gt_sorted
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)

class LovaszHinge(nn.Module):
    def forward(self, logits, y):
        logit_fg = logits[:, 1].contiguous().view(-1)
        y = y.contiguous().view(-1)
        return lovasz_hinge_flat(logit_fg, y)

# --- registry for training loop ---
LOSS_BANK = {
    "bce_weight":   weighted_bce,          # weighted cross-entropy (baseline)
    "dice":         DiceLoss,              # overlap-focused, ignores bg ratio
    "combo":        ComboLoss,             # default α=0.5
    "focal_tversky":FocalTverskyLoss,      # tune α,β,γ as needed
    "lovasz":       LovaszHinge            # direct IoU optimisation
}

# usage:
# criterion = LOSS_BANK["combo"](alpha=0.5)
# loss = criterion(logits, y)
class CombinedLoss(nn.Module):
    """
    Build a weighted sum of arbitrary losses listed in LOSS_BANK.
    Example:   criterion = CombinedLoss(
                   ["bce_weight", "dice", "focal_tversky"],
                   weights=[0.4, 0.4, 0.2])
    """
    def __init__(self, names, weights=None, **kwargs):
        super().__init__()
        assert all(n in LOSS_BANK for n in names), "unknown loss key"
        self.loss_fns = nn.ModuleList([LOSS_BANK[n](**kwargs) if callable(LOSS_BANK[n]) else LOSS_BANK[n] for n in names])
        w = torch.tensor(weights) if weights is not None else torch.ones(len(names))
        self.weights = (w / w.sum()).tolist()                 # normalise

    def forward(self, logits, y):
        total = 0.
        for w, fn in zip(self.weights, self.loss_fns):
            total += w * fn(logits, y)
        return total
# 1) Classic Dice + weighted BCE (50-50)
criterion = CombinedLoss(["bce_weight", "dice"])

# 2) Heavier Dice emphasis (70 % Dice, 30 % BCE)
criterion = CombinedLoss(["dice", "bce_weight"], weights=[0.7, 0.3])

# 3) Tackle noisy labels: small focal-Tversky term
criterion = CombinedLoss(["combo", "focal_tversky"], weights=[0.8, 0.2], alpha=0.5)

# Section 9 · Training Engine
# All knobs controlled via CFG – change a string, keep the loop intact.

import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import (OneCycleLR, CosineAnnealingWarmRestarts,
                                      CyclicLR, ReduceLROnPlateau)
from copy import deepcopy
try: import torch_optimizer as topt               # extra optims (radam, lookahead)
except ImportError: topt = None

# ───────────────────────── CFG additions ─────────────────────────
CFG.update({
    "epochs"   : 20,
    "optim"    : "adamw",         # adamw · radam · sgd · lookahead_sgd
    "sched"    : "onecycle",      # onecycle · cosine · cyclic · plateau
    "lr"       : 5e-4,
    "grad_clip": 1.0,             # None = off   / keeps large spikes stable
    "ema_decay": 0.99,            # None = off   / 0.99 works well for radar
})

# ───────────────────── optimiser & scheduler builders ────────────
def build_optimizer(model):
    lr = CFG["lr"]; wd = 1e-2
    if CFG["optim"] == "adamw":
        return optim.AdamW(model.parameters(), lr, weight_decay=wd)
    if CFG["optim"] == "sgd":
        return optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=wd)
    if CFG["optim"] == "radam" and topt:
        return topt.RAdam(model.parameters(), lr, weight_decay=wd)          # variance-rectified Adam
    if CFG["optim"] == "lookahead_sgd" and topt:
        base = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=wd)
        return topt.Lookahead(base)                                         # slower but very stable
    raise ValueError("Unsupported optimiser in CFG['optim']")

def build_scheduler(opt, steps_per_epoch):
    if CFG["sched"] == "onecycle":
        # good default – large LR peak then anneal
        return OneCycleLR(opt, max_lr=CFG["lr"], total_steps=CFG["epochs"]*steps_per_epoch)
    if CFG["sched"] == "cosine":
        # gradually restarts LR – useful for long runs
        return CosineAnnealingWarmRestarts(opt, T_0=steps_per_epoch*4)
    if CFG["sched"] == "cyclic":
        # bounce between two LRs – speeds small-set adaptation
        return CyclicLR(opt, base_lr=CFG["lr"]/10, max_lr=CFG["lr"],
                        step_size_up=steps_per_epoch*2, cycle_momentum=False)
    if CFG["sched"] == "plateau":
        # cut LR when val metric stalls – handy for fine-tune scripts
        return ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    return None

# ───────────────────────── EMA helper ────────────────────────────
class EMA:
    """Exponential Moving Average – smoother weights; boosts val stability."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
    def apply(self, model):
        self.backup = deepcopy(model.state_dict())
        model.load_state_dict(self.shadow, strict=False)
    def restore(self, model):
        model.load_state_dict(self.backup, strict=False)

# ───────────────────────── training loop ─────────────────────────
def train(model, loader_tr, loader_val, criterion,
          metric_name="weighted", patience=5):
    device = CFG["device"]; model.to(device)
    opt   = build_optimizer(model)
    sched = build_scheduler(opt, len(loader_tr))
    scaler= GradScaler()                                # AMP scaler
    ema   = EMA(model, CFG["ema_decay"]) if CFG["ema_decay"] else None
    metric_fn = METRICS[metric_name]["fn"]

    best, bad = -1, 0
    for ep in range(CFG["epochs"]):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            xb, yb = apply_train_aug(xb, yb)             # Section 2 aug bag
            with autocast():
                logits = model(xb)
                loss   = criterion(logits, yb)
            scaler.scale(loss).backward()
            if CFG["grad_clip"]:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt); scaler.update(); opt.zero_grad()
            if sched and not isinstance(sched, ReduceLROnPlateau): sched.step()
            if ema: ema.update(model)

        # ─ validation ─
        if ema: ema.apply(model)
        model.eval(); vals = []
        with torch.no_grad(), autocast():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                vals.append(metric_fn(model(xb).cpu(), yb))
        val = sum(vals) / len(vals)
        if ema: ema.restore(model)
        if sched and isinstance(sched, ReduceLROnPlateau): sched.step(val)

        print(f"ep{ep:02d} val {val:.4f}")
        if val > best:
            best, bad = val, 0
            torch.save(model.state_dict(), "ckpt_best.pth")
        else:
            bad += 1
            if bad >= patience:
                print("early stop"); break

# -----------------------------------------------------------------
# Quick flipping guide
#   Heavy over-fitting         → set optim='sgd', sched='cyclic'
#   Val metric flat            → sched='plateau', increase patience
#   Training unstable spikes   → grad_clip=0.5, optim='radam'
#   Tiny adaptation set onsite → epochs=5, sched='cyclic', optim='adamw'

# Section 10 · Inference + Test-Time Augmentation
# Usage
#   preds = infer_batch(model, x)            # single batch → B×H×W masks
#   run_inference(model, loader, tta=True)   # full loader → dict{id:mask}

import torch, torch.nn.functional as F
import json, numpy as np

# load saved threshold if any (for Dice/IoU metrics)
try:
    _THR_JSON = json.load(open("best_thr.json"))
    _THR = _THR_JSON.get("thr", 0.5); _METRIC = _THR_JSON.get("metric", "")
except FileNotFoundError:
    _THR = 0.5; _METRIC = ""

# simple horizontal flip TTA (azimuth symmetry)
def _tta_logits(model, x):
    logits = model(x)
    logits_flip = torch.flip(model(torch.flip(x, [-1])), [-1])
    return (logits + logits_flip) / 2

def infer_batch(model, x, tta=False):
    model.eval()
    with torch.no_grad():
        logits = _tta_logits(model, x) if tta else model(x)
        if CFG["out_ch"] == 2:
            # binary segmentation
            probs = F.softmax(logits, 1)[:, 1]
            thr = _THR if _METRIC in ("dice", "iou") else 0.5
            return (probs > thr).long()
        else:
            # multi-class argmax
            return logits.argmax(1)

def run_inference(model, loader, tta=False, return_dict=False):
    """
    Returns a tensor of masks or a dict {idx:mask} for submission packing.
    Assumes loader.dataset is indexable so we can map batch index to file.
    """
    masks, ids = [], []
    for i, (x, _) in enumerate(loader):
        x = x.to(CFG["device"])
        masks.append(infer_batch(model, x, tta).cpu())
        ids.extend(range(i * CFG["batch"], i * CFG["batch"] + x.size(0)))
    masks = torch.cat(masks)
    return dict(zip(ids, masks)) if return_dict else masks

# Section 11 · Few-Shot Adaptation Script
# Fine-tunes a base checkpoint on a small labelled set the organisers might provide.
#   • freeze_encoder=True  keeps backbone frozen → faster & avoids over-fit
#   • epochs, lr, sched come from CFG but we set light defaults

def adapt_fewshot(base_ckpt,
                  small_train_dir,
                  freeze_encoder=True,
                  out_ckpt="adapted.pth"):
    # build model exactly as in training
    model = ResUNetCBAM(in_ch=CFG["in_ch"], out_ch=CFG["out_ch"], use_attn=True)
    model.load_state_dict(torch.load(base_ckpt, map_location=CFG["device"]))
    if freeze_encoder:                       # only decoder & head learn
        for n, p in model.named_parameters():
            if not n.startswith("up") and not n.startswith("head"):
                p.requires_grad_(False)
    # tiny LR & epoch count
    CFG.update({"lr": 3e-4, "epochs": 5, "sched": "plateau"})
    crit = LOSS_BANK["combo"](alpha=0.5)
    loader_tr = make_loader(small_train_dir, "train", batch=CFG["batch"])
    loader_va = make_loader(small_train_dir, "val",   batch=CFG["batch"])
    train(model, loader_tr, loader_va, crit, metric_name="weighted")
    torch.save(model.state_dict(), out_ckpt)
    print("few-shot adaptation done →", out_ckpt)
</code></pre>

</details>
