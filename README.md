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

<details>
<summary>Best use cases (click to expand)</summary>

<pre><code class="language-python">
# Section 1 · Weather V1  ―  At-Home Solution (fully annotated)
# “### ADD” marks every line / block that diverges from the organiser baseline.

# Imports
import math, random, datetime as dt
from pathlib import Path, PurePath
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.data as td
from torch.cuda.amp import autocast, GradScaler

# Config (baseline kept only bands/batch/device)
CFG = dict(
    bands      = 16,
    batch      = 32,
    epochs     = 30,           ### ADD  – baseline had 10
    lr         = 5e-4,         ### ADD  – adamw instead of sgd 1e-3
    device     = "cuda" if torch.cuda.is_available() else "cpu",
    pos_weight = None,         ### ADD  – filled by calibrate_pos_weight()
    focal_alpha= None,         ### ADD  – set alongside pos_weight
    thr        = 0.4           ### ADD  – tuned Dice/Acc threshold
)

# ---------------------- dataset + metadata ----------------------
def sun_elev(lat, lon, utc):
    jd = utc.timetuple().tm_yday + utc.hour/24
    decl = 23.44*math.cos(math.radians((jd+10)*360/365))
    ha   = (utc.hour*15 + lon) - 180
    elev = math.asin(
        math.sin(math.radians(lat))*math.sin(math.radians(decl)) +
        math.cos(math.radians(lat))*math.cos(math.radians(decl))*math.cos(math.radians(ha))
    )
    return math.sin(elev)      # −1 … 1

def load_pt(p: Path):
    t = torch.load(p)          # 17×H×W  (16 bands + mask)
    x, y = t[:-1].float(), t[-1].long()
    # metadata from filename “…_lat13.5_lon102.3_20250815T1710.pt”
    lat = float(PurePath(p).stem.split("_lat")[1].split("_")[0])
    lon = float(PurePath(p).stem.split("_lon")[1].split("_")[0])
    utc = dt.datetime.strptime(PurePath(p).stem.split("_")[-1], "%Y%m%dT%H%M")
    meta = torch.tensor([lat/90, lon/180, sun_elev(lat, lon, utc)], dtype=torch.float32)
    return x, y, meta

class SatDS(td.Dataset):
    def __init__(self, root, split):
        self.files = sorted(Path(root, split).glob("*.pt"))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):  return load_pt(self.files[i])

def collate(batch):
    xs, ys, ms = zip(*batch)
    xs, ys, ms = torch.stack(xs), torch.stack(ys), torch.stack(ms)
    CFG["img_shape"] = xs.shape[-2:]
    return xs, ys, ms

def make_loader(root, split):
    return td.DataLoader(SatDS(root, split),
                         batch_size=CFG["batch"],
                         shuffle=(split=="train"),
                         collate_fn=collate)

# -------------------------- augmentation ------------------------
def random_band_drop(x, p=0.2):          ### ADD
    if random.random() < p:
        x[:, torch.randint(0, x.size(1), ())] = 0
    return x
def add_noise(x, σ=0.01): return x + σ*torch.randn_like(x)   ### ADD
def hor_flip(x, y):
    if random.random() < 0.5:
        x = torch.flip(x, [-1]); y = torch.flip(y, [-1])
    return x, y
def aug(x, y):
    x = random_band_drop(x); x = add_noise(x)
    x, y = hor_flip(x, y)
    return x, y

# ------------------------ model ---------------------------------
class ResBlock(nn.Module):               ### ADD (InstanceNorm + 2d-Dropout)
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.InstanceNorm2d(c); self.n2 = nn.InstanceNorm2d(c)
        self.c1 = nn.Conv2d(c,c,3,1,1); self.c2 = nn.Conv2d(c,c,3,1,1)
        self.drop, self.act = nn.Dropout2d(0.1), nn.SiLU()
    def forward(self, x):
        h = self.act(self.n1(x)); h = self.drop(self.c1(h))
        h = self.act(self.n2(h)); h = self.c2(h)
        return x + h

class FiLM(nn.Module):                   ### ADD – scalar conditioning
    def __init__(self, ch, cond=3, hid=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond,hid), nn.ReLU(),
                                 nn.Linear(hid,ch*2))
    def forward(self, f, meta):
        γβ = self.mlp(meta); γ, β = γβ.chunk(2,1)
        return f*(1+γ.view(-1,f.size(1),1,1)) + β.view(-1,f.size(1),1,1)

class UNetV1(nn.Module):
    def __init__(self, in_ch=16, use_attn=False):   ### use_attn reserved for CBAM
        super().__init__()
        self.stem = nn.Conv2d(in_ch,64,3,1,1)
        self.d1 = nn.Conv2d(64,128,4,2,1); self.rb1=ResBlock(128)
        self.d2 = nn.Conv2d(128,256,4,2,1); self.rb2=ResBlock(256)
        self.d3 = nn.Conv2d(256,512,4,2,1); self.rb3=ResBlock(512)
        self.mid = ResBlock(512)
        self.film= FiLM(512,3)
        self.u3 = nn.ConvTranspose2d(512,256,4,2,1)
        self.u2 = nn.ConvTranspose2d(512,128,4,2,1)
        self.u1 = nn.ConvTranspose2d(256,64,4,2,1)
        self.out= nn.Conv2d(128,1,1)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(), nn.Linear(512,1))
    def forward(self, x, meta):
        s  = self.stem(x)
        d1 = self.rb1(self.d1(s))
        d2 = self.rb2(self.d2(d1))
        d3 = self.rb3(self.d3(d2))
        m  = self.film(self.mid(d3), meta)
        u3 = self.u3(m)
        u2 = self.u2(torch.cat([u3,d2],1))
        u1 = self.u1(torch.cat([u2,d1],1))
        mask = self.out(torch.cat([u1,s],1))
        flag = self.cls(m).squeeze(1)
        return mask, flag

# -------------------- dynamic class weight + focal α ------------
def calibrate_pos_weight(loader_tr):     ### ADD
    fg, px = 0, 0
    for _, y, _ in loader_tr:
        fg += y.sum().item(); px += y.numel()
    p = fg / px
    CFG["pos_weight"] = (1-p)/p
    CFG["focal_alpha"]= 1 - p            # FG weight in focal loss
    print(f"class-imbalance p={p:.3%}  pos_w={CFG['pos_weight']:.4f}  α={CFG['focal_alpha']:.4f}")

# --------------------------- losses ------------------------------
class DiceLoss(nn.Module):
    def forward(self, logit, y):
        p = torch.sigmoid(logit)
        inter = (p*y).sum(); union = p.sum()+y.sum()
        return 1 - (2*inter+1)/(union+1)

class FocalLoss(nn.Module):             ### ADD – uses calibrated α
    def __init__(self, γ=2):
        super().__init__(); self.γ=γ
    def forward(self, logit, y):
        α = CFG["focal_alpha"]
        p  = torch.sigmoid(logit)
        pt = p*y + (1-p)*(1-y)
        w  = α*y + (1-α)*(1-y)
        return (w*((1-pt)**self.γ)*(-pt.log())).mean()

def active_contour(logit, y, λ=1, μ=1):
    p = torch.sigmoid(logit)
    dy, dx = torch.gradient(p, dim=(2,3))
    length = torch.sqrt((dx**2+dy**2)+1e-8).mean()
    region = (λ*((p-y)**2)*y + μ*((p-y)**2)*(1-y)).mean()
    return length+region

class WeatherLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(); self.focal = FocalLoss()
    def forward(self, mask_logit, img_logit, y):
        flag = (y.sum((1,2))>0).float()
        seg = 0.5*self.focal(mask_logit,y) + 0.3*self.dice(mask_logit,y) + \
              0.2*active_contour(mask_logit,y)
        img = F.binary_cross_entropy_with_logits(img_logit, flag)
        return seg + 0.2*img

# ------------------------ metric -------------------------------
def dice_acc(mask_logit, img_logit, y, thr=CFG["thr"]):
    dice = 1 - DiceLoss()(mask_logit, y)
    pred_flag = (torch.sigmoid(img_logit)>0.5).long()
    acc = (pred_flag == (y.sum((1,2))>0)).float().mean()
    return 0.5*(dice+acc)

# ------------------------ training -----------------------------
def train(root):
    tr = make_loader(root,"train")
    calibrate_pos_weight(tr)            # sets pos_weight & α
    va = make_loader(root,"val")
    net=UNetV1(CFG["bands"]).to(CFG["device"])
    opt=torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=1e-2)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"],
                           total_steps=len(tr)*CFG["epochs"])
    scaler, criterion = GradScaler(), WeatherLoss()
    best = 0
    for ep in range(CFG["epochs"]):
        net.train()
        for x,y,m in tr:
            x,y,m = x.to(CFG["device"]),y.to(CFG["device"]),m.to(CFG["device"])
            x,y = aug(x,y)
            with autocast():
                mask,flag = net(x,m)
                loss = criterion(mask,flag,y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        net.eval(); vals=[]
        with torch.no_grad(), autocast():
            for x,y,m in va:
                x,y,m = x.to(CFG["device"]),y.to(CFG["device"]),m.to(CFG["device"])
                mask,flag = net(x,m)
                vals.append(dice_acc(mask,flag,y))
        val = torch.tensor(vals).mean().item()
        if val>best: best=val; torch.save(net.state_dict(),"weather_best.pth")
        print(f"ep{ep:02d} val {val:.4f}")

# call training:
# train("/path/to/weather_dataset")

# Section 2 · Config + Loader + Meta-parser
# This chunk covers S-0 and S-1 in one go.
# Drop it at the top of your satellite notebook.

import math, datetime as dt
from pathlib import Path, PurePath
import torch, torch.utils.data as td

# ---------- CFG – edit once, everything downstream picks it up ----------
CFG = dict(
    # data-specific
    bands       = 16,          # change if organiser adds/removes channels
    img_shape   = None,        # auto-filled after first batch
    use_meta    = True,        # lat/lon/UTC → sun-elev conditioning
    metric_name = "dice_acc",  # default competition metric
    # training hyper-params (overridden later if needed)
    batch       = 32,
    epochs      = 30,
    lr          = 5e-4,
    device      = "cuda" if torch.cuda.is_available() else "cpu",
    # class-imbalance weights (filled by calibrate_pos_weight later)
    pos_weight  = None,
    focal_alpha = None,
    thr         = 0.4          # initial mask threshold for Dice/Acc blend
)

# ---------- tiny helper: sun elevation normalised to −1 … 1 ----------
def sun_elev(lat, lon, utc):
    jd   = utc.timetuple().tm_yday + utc.hour / 24
    decl = 23.44 * math.cos(math.radians((jd + 10) * 360 / 365))
    ha   = (utc.hour * 15 + lon) - 180          # hour angle
    elev = math.asin(
        math.sin(math.radians(lat)) * math.sin(math.radians(decl)) +
        math.cos(math.radians(lat)) * math.cos(math.radians(decl)) *
        math.cos(math.radians(ha))
    )
    return math.sin(elev)                       # −1…1

# ---------- file → (x, y, meta)  loader ----------------------------------
def load_pt(f: Path):
    """
    Expects organiser .pt with 17×H×W tensor:
        16 bands (float32) + 1 binary mask (int64)
    Filename carries metadata:
        “…_lat13.5_lon102.3_20250815T1710.pt”
    Returns:
        x : 16×H×W  float32   (NaNs replaced by 0)
        y :    H×W  int64
        m : 3-dim meta tensor  (lat, lon, sun-elev)
    """
    t   = torch.load(f)                         # shape 17×H×W
    x   = t[:-1].float()
    x[torch.isnan(x)] = 0                       # NaNs → 0  (handles band gaps)
    y   = t[-1].long()

    if CFG["use_meta"]:
        lat = float(PurePath(f).stem.split("_lat")[1].split("_")[0])
        lon = float(PurePath(f).stem.split("_lon")[1].split("_")[0])
        utc = dt.datetime.strptime(PurePath(f).stem.split("_")[-1], "%Y%m%dT%H%M")
        meta = torch.tensor([lat / 90, lon / 180, sun_elev(lat, lon, utc)],
                            dtype=torch.float32)
    else:
        meta = torch.zeros(3)

    return x, y, meta

# ---------- torch Dataset / DataLoader -----------------------------------
class SatDataset(td.Dataset):
    def __init__(self, root, split):
        self.files = sorted(Path(root, split).glob("*.pt"))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return load_pt(self.files[idx])

def collate_fn(batch):
    xs, ys, ms = zip(*batch)
    xs, ys, ms = torch.stack(xs), torch.stack(ys), torch.stack(ms)
    # remember true image shape for dynamic padding later
    CFG["img_shape"] = xs.shape[-2:]
    return xs, ys, ms

def make_loader(root, split):
    """
    root/
      └── train/*.pt
          val/*.pt
    """
    return td.DataLoader(SatDataset(root, split),
                         batch_size=CFG["batch"],
                         shuffle=(split == "train"),
                         collate_fn=collate_fn)
# Section 3 · Augmentation Bag (Spectral & Geometric)
# Hook into training with:   x, y = apply_sat_aug(x, y)
# Each aug is a plain function so you can reorder or comment out lines.

import torch, random, torch.nn.functional as F

# -------- band-level corruption ---------------------------------
def random_band_drop(x, p=0.2):
    """
    Zero-out one spectral band with prob p.
    Guards against real validation files where a VIS/IR channel is missing.
    """
    if random.random() < p:
        band = torch.randint(0, x.size(1), ())
        x[:, band] = 0
    return x

def gaussian_noise(x, sigma=0.01):
    """
    Additive Gaussian noise – covers sensor SNR shifts or compression artefacts.
    """
    return x + sigma * torch.randn_like(x)

# -------- sample-level blending ---------------------------------
def mixup(x, y, alpha=0.4):
    """
    MixUp for segmentation: convex combination of two images.
    Only use when your loss can handle soft labels (e.g. BCE/Dice).
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    # keep hard label of dominant sample for simplicity
    y_mix = y if lam >= 0.5 else y[idx]
    return x_mix, y_mix

def cutmix_patch(x, y, alpha=1.0, max_prop=0.4):
    """
    CutMix – paste random patch from another image in the batch.
    More aggressive than MixUp; good against over-fitting when train < val size.
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B, C, H, W = x.size()
    cut_w = int(W * max_prop * lam ** 0.5)
    cut_h = int(H * max_prop * lam ** 0.5)
    cx, cy = torch.randint(0, W, (1,)), torch.randint(0, H, (1,))
    x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
    idx = torch.randperm(B)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    y[:,   y1:y2, x1:x2] = y[idx,   y1:y2, x1:x2]
    return x, y

# -------- geometric invariance ---------------------------------
def horizontal_flip(x, y):
    """
    Flip along longitude – valid because physical latitude order stays.
    Disable if organiser’s task attaches absolute longitudes to classes!
    """
    if random.random() < 0.5:
        x = torch.flip(x, [-1]); y = torch.flip(y, [-1])
    return x, y

# -------- master switchboard -----------------------------------
def apply_sat_aug(x, y, *, use_mixup=False, use_cutmix=False):
    """
    Call inside training loop *before* sending to model.
      x, y = apply_sat_aug(x, y)
    Toggle MixUp / CutMix via kwargs.
    """
    x = random_band_drop(x)
    x = gaussian_noise(x)
    x, y = horizontal_flip(x, y)
    if use_mixup:
        x, y = mixup(x, y)
    if use_cutmix:
        x, y = cutmix_patch(x, y)
    return x, y
# Section 4 · Metadata Conditioning Blocks  (S-3)
# Plug the returned module into your UNet bottleneck:
#
#     self.meta_mod = build_meta_block(feat_ch=512,
#                                      mode="film",      # or "channel"
#                                      cond_dim=3)       # length of meta vector
#     ...
#     feats = self.meta_mod(feats, meta_vec)
#
# Modes
#   "film"     – FiLM γ/β modulation   (default, good for small scalar vectors)
#   "channel"  – Channel attention     (sigmoid weights)   use when meta vector
#                should softly gate each feature map.

import torch, torch.nn as nn

# ---------- core primitives ------------------------------------
class FiLM(nn.Module):
    """Feature-wise linear modulation: feats * (1+γ) + β."""
    def __init__(self, feat_ch, cond_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_ch * 2)
        )
    def forward(self, feats, meta):
        γβ = self.net(meta)               # B×2C
        γ, β = γβ.chunk(2, dim=1)
        γ = γ.view(-1, feats.size(1), 1, 1)
        β = β.view(-1, feats.size(1), 1, 1)
        return feats * (1 + γ) + β

class ChannelAttention(nn.Module):
    """Apply sigmoid gate per feature map: feats * σ(Meta→C)."""
    def __init__(self, feat_ch, cond_dim, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_ch),  nn.Sigmoid()
        )
    def forward(self, feats, meta):
        g = self.fc(meta).view(-1, feats.size(1), 1, 1)
        return feats * g

# ---------- factory helper ------------------------------------
def build_meta_block(feat_ch, mode="film", cond_dim=3, hidden=64):
    """
    mode : "film" | "channel" | None
    cond_dim : length of meta vector (e.g. 3 if [lat, lon, sun])
    """
    if mode is None or cond_dim == 0:
        return nn.Identity()
    if mode == "film":
        return FiLM(feat_ch, cond_dim, hidden)
    if mode == "channel":
        return ChannelAttention(feat_ch, cond_dim, hidden)
    raise ValueError(f"Unknown meta conditioning mode: {mode}")
# How to use Inside your UNet bottleneck
self.meta_mod = build_meta_block(feat_ch=512,
                                 mode="film",      # or "channel", or None
                                 cond_dim=meta.size(1))

...
feats = self.meta_mod(feats, meta)   # meta is B×cond_dim

# Section 5 · Backbone — Residual UNet (InstanceNorm, Dropout2d, optional CBAM)
# This is the *conventional, explicit* layer-by-layer version —
# no loops, no dynamic padding, mirrors the style you used in Weather V1.

import torch, torch.nn as nn

# ------------------------------------------------------------
# Optional CBAM attention  (channel- & spatial)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Residual Conv Block:   IN → CONV → Drop2d → IN → CONV (+ res) → CBAM?
# ------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, dropout_p=0.1, use_attn=False):
        super().__init__()
        self.n1 = nn.InstanceNorm2d(ch)
        self.n2 = nn.InstanceNorm2d(ch)
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.drop, self.act = nn.Dropout2d(dropout_p), nn.SiLU()
        self.attn = CBAM(ch) if use_attn else nn.Identity()

    def forward(self, x):
        h = self.act(self.n1(x))
        h = self.drop(self.c1(h))
        h = self.act(self.n2(h))
        h = self.c2(h)
        return self.attn(x + h)

# ------------------------------------------------------------
# UNet-Res backbone, depth = 4 (handles up to 256 × 256 cleanly)
#   • meta_block: any nn.Module(feats, meta)  (FiLM, channel-attn …)
# ------------------------------------------------------------
class UNetRes(nn.Module):
    """
    Args
    ----
    in_ch      : # input spectral bands (e.g. 16)
    base       : # filters after stem (64 default)
    dropout_p  : 2-D dropout prob inside every ResBlock
    use_attn   : True → wrap each ResBlock with CBAM
    meta_block : optional conditioning module; Identity if None
    """
    def __init__(self, in_ch, base=64,
                 dropout_p=0.1, use_attn=False,
                 meta_block=None):
        super().__init__()
        # Stem
        self.stem = nn.Conv2d(in_ch, base, 3, 1, 1)

        # Encoder
        self.down1 = nn.Conv2d(base, base*2, 4, 2, 1)
        self.rb1   = ResBlock(base*2, dropout_p, use_attn)

        self.down2 = nn.Conv2d(base*2, base*4, 4, 2, 1)
        self.rb2   = ResBlock(base*4, dropout_p, use_attn)

        self.down3 = nn.Conv2d(base*4, base*8, 4, 2, 1)
        self.rb3   = ResBlock(base*8, dropout_p, use_attn)

        # Bottleneck
        self.mid   = ResBlock(base*8, dropout_p, use_attn)
        self.meta  = meta_block if meta_block is not None else nn.Identity()

        # Decoder
        self.up3   = nn.ConvTranspose2d(base*8, base*4, 4, 2, 1)
        self.cv3   = nn.Conv2d(base*8, base*4, 3, 1, 1)

        self.up2   = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.cv2   = nn.Conv2d(base*4, base*2, 3, 1, 1)

        self.up1   = nn.ConvTranspose2d(base*2, base,   4, 2, 1)
        self.cv1   = nn.Conv2d(base*2, base,   3, 1, 1)

        self.out_channels = base     # expose for head attachment
        self.tail = nn.Conv2d(base*2, base, 3, 1, 1)   # concat stem skip later

    def forward(self, x, meta=None):
        s0 = self.stem(x)            # B × base × H × W

        d1 = self.rb1(self.down1(s0))  # B × 2B × H/2
        d2 = self.rb2(self.down2(d1))  # B × 4B × H/4
        d3 = self.rb3(self.down3(d2))  # B × 8B × H/8

        bott = self.meta(self.mid(d3), meta)            # apply FiLM if any

        u3 = self.up3(bott)                             # H/4
        u3 = self.cv3(torch.cat([u3, d2], 1))

        u2 = self.up2(u3)                               # H/2
        u2 = self.cv2(torch.cat([u2, d1], 1))

        u1 = self.up1(u2)                               # H
        u1 = self.cv1(torch.cat([u1, s0], 1))

        feats = self.tail(torch.cat([u1, s0], 1))       # final feature map
        return feats
# Section 6 · Heads (S-5)
# Attach exactly one of these heads to the backbone’s final feature map.
#
# Usage example
# -------------
#     feats = backbone(x, meta)           # B × C × H × W
#     head   = build_head(feat_ch=backbone.out_channels,
#                         head_type="binary",    # binary | multi | reg
#                         num_classes=3)         # used for head_type="multi"
#     logits = head(feats)
#
# Available head types
#   • "binary"   – 1-channel sigmoid mask  (+ optional image-flag)
#   • "multi"    – N-channel softmax mask
#   • "reg"      – 1-channel rain-rate regression (mm h⁻¹)
# If you need both pixel mask *and* image-flag, set `include_flag=True`.

import torch, torch.nn as nn

# ---------------- basic pixel heads ------------------------------------
class BinaryMaskHead(nn.Module):
    def __init__(self, in_ch): super().__init__(); self.conv = nn.Conv2d(in_ch, 1, 1)
    def forward(self, feats):  return self.conv(feats)          # logits

class MultiClassHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, 1)
    def forward(self, feats):  return self.conv(feats)          # logits

class RegressionHead(nn.Module):
    def __init__(self, in_ch): super().__init__(); self.conv = nn.Conv2d(in_ch, 1, 1)
    def forward(self, feats):  return self.conv(feats)          # linear value

# ---------------- optional image-level flag ----------------------------
class ImageFlag(nn.Module):
    """Global rain / no-rain flag via GAP + FC."""
    def __init__(self, in_ch):
        super().__init__()
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 1)
        )
    def forward(self, feats):  return self.cls(feats).squeeze(1)

# ---------------- factory helper --------------------------------------
class HeadWrapper(nn.Module):
    """
    Returns a tuple:   (pixel_output, img_flag or None)
    pixel_output shape:
        binary, reg    → B×1×H×W
        multi          → B×num_classes×H×W
    """
    def __init__(self, feat_ch, head_type="binary", num_classes=2,
                 include_flag=False):
        super().__init__()
        if head_type == "binary":
            self.pix = BinaryMaskHead(feat_ch)
        elif head_type == "multi":
            self.pix = MultiClassHead(feat_ch, num_classes)
        elif head_type == "reg":
            self.pix = RegressionHead(feat_ch)
        else:
            raise ValueError("head_type must be binary | multi | reg")

        self.flag = ImageFlag(feat_ch) if include_flag else None

    def forward(self, feats):
        pixel = self.pix(feats)
        fflag = self.flag(feats) if self.flag else None
        return pixel, fflag

def build_head(feat_ch, head_type="binary", num_classes=2, include_flag=True):
    """
    Convenience wrapper:
        head = build_head(backbone.out_channels, "binary", include_flag=True)
    """
    return HeadWrapper(feat_ch, head_type, num_classes, include_flag)
# Section 7 · Loss Bank & Mixer  (S-6)
# Each loss takes logits + ground-truth mask (plus img_flag when needed).
# Combine any subset with one line:
#
#   criterion = CombinedLoss(
#       names   = ["focal", "dice", "contour", "flag_bce"],
#       weights = [0.4,      0.3,    0.1,      0.2]   # auto-normalised
#   )
#
# Dynamic foreground weight & focal α are read from CFG, set once by
# `calibrate_pos_weight(loader_tr)` (see Section 2).

import torch, torch.nn as nn, torch.nn.functional as F

# ---------------- individual pixel losses ---------------------------------
class DiceLoss(nn.Module):
    def forward(self, logit, y):
        p = torch.sigmoid(logit)
        inter = (p*y).sum(); union = p.sum()+y.sum()
        return 1 - (2*inter+1)/(union+1)

class FocalLoss(nn.Module):
    def __init__(self, γ=2):
        super().__init__(); self.γ = γ
    def forward(self, logit, y):
        α = CFG["focal_alpha"]         # set by calibrate_pos_weight
        p  = torch.sigmoid(logit)
        pt = p*y + (1-p)*(1-y)
        w  = α*y + (1-α)*(1-y)
        return (w * (1-pt).pow(self.γ) * (-pt.log())).mean()

def active_contour(logit, y, λ=1, μ=1):
    p = torch.sigmoid(logit)
    dy, dx = torch.gradient(p, dim=(2,3))
    length = torch.sqrt((dx**2 + dy**2) + 1e-8).mean()
    region = (λ*((p-y)**2)*y + μ*((p-y)**2)*(1-y)).mean()
    return length + region

class TverskyLoss(nn.Module):
    def __init__(self, α=0.7, β=0.3):
        super().__init__(); self.a, self.b = α, β
    def forward(self, logit, y):
        p = torch.sigmoid(logit)
        tp = (p*y).sum(); fp = (p*(1-y)).sum(); fn = ((1-p)*y).sum()
        return 1 - (tp + 1) / (tp + self.a*fp + self.b*fn + 1)

# Lovász hinge surrogate for IoU (binary)
def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - inter / union
    jaccard[1:] -= jaccard[:-1]
    return jaccard

def lovasz_binary_flat(logits, labels):
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)

class LovaszHinge(nn.Module):
    def forward(self, logit, y):
        return lovasz_binary_flat(logit.view(-1), y.view(-1))

# image-level flag BCE
class FlagBCE(nn.Module):
    def forward(self, img_logit, y_mask):
        flag = (y_mask.sum((1,2)) > 0).float()
        return F.binary_cross_entropy_with_logits(img_logit, flag)

# ---------------- registry -----------------------------------------------
LOSS_BANK = {
    "focal"    : FocalLoss,
    "dice"     : DiceLoss,
    "contour"  : lambda: active_contour,    # functional form
    "tversky"  : TverskyLoss,
    "lovasz"   : LovaszHinge,
    "flag_bce" : FlagBCE
}

# ---------------- mixer ---------------------------------------------------
class CombinedLoss(nn.Module):
    """
    names   : list of keys from LOSS_BANK.
    weights : same length; will be re-normalised.
    Example:
        criterion = CombinedLoss(["focal","dice","flag_bce"],
                                 [0.5,    0.3,   0.2])
    """
    def __init__(self, names, weights):
        super().__init__()
        assert len(names) == len(weights) and all(n in LOSS_BANK for n in names)
        # convert weights → tensor & normalise
        w = torch.tensor(weights, dtype=torch.float)
        self.weights = (w / w.sum()).tolist()
        # instantiate or keep callable
        self.loss_fns = []
        for n in names:
            lf = LOSS_BANK[n]()
            self.loss_fns.append(lf)

    def forward(self, mask_logit, img_logit, y):
        total = 0.
        for w, fn in zip(self.weights, self.loss_fns):
            if isinstance(fn, nn.Module):
                # pixel-wise loss
                loss = fn(mask_logit, y) if not isinstance(fn, FlagBCE) \
                       else fn(img_logit, y)
            else:
                # functional contour loss
                loss = fn(mask_logit, y)
            total += w * loss
        return total
# Classic V1 mix  (Focal + Dice + Contour + Flag)
crit = CombinedLoss(["focal","dice","contour","flag_bce"],
                    [0.4,   0.3,  0.1,      0.2])

# Metric flips to IoU only
crit = CombinedLoss(["lovasz"], [1.0])

# Heavy FP penalty scenario
crit = CombinedLoss(["tversky","flag_bce"], [0.8, 0.2])
# Section 8 · Training Engine  (build → train → validate → save best)
# One entry-point:    train_sat(root_dir)
# Flags live in CFG (optim, sched, epochs, grad_clip, etc.) so you flip
# behaviour without rewriting the loop.

import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import (OneCycleLR, CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau)

# ------------ optimiser / scheduler builders -----------------------------
def build_optimizer(model):
    lr = CFG["lr"]; wd = 1e-2
    return optim.AdamW(model.parameters(), lr, weight_decay=wd)  # good default

def build_scheduler(opt, steps_per_epoch):
    return OneCycleLR(opt, max_lr=CFG["lr"],
                      total_steps=steps_per_epoch * CFG["epochs"])

# ------------ training loop ----------------------------------------------
def train_sat(root):
    # 1) loaders
    loader_tr = make_loader(root, "train")        # from Section 2
    calibrate_pos_weight(loader_tr)               # sets pos_weight & focal α
    loader_va = make_loader(root, "val")

    # 2) build model = backbone + head
    meta_mod  = build_meta_block(512, mode="film", cond_dim=3)           # Sec 4
    backbone  = UNetRes(in_ch=CFG["bands"],
                        dropout_p=0.1,
                        use_attn=True,
                        meta_block=meta_mod)                             # Sec 5
    head      = build_head(backbone.out_channels,
                           head_type="binary",
                           include_flag=True)                            # Sec 6
    model     = nn.Sequential(backbone, head)                            # simple wrap

    model.to(CFG["device"])

    # 3) loss, optim, sched, AMP scaler
    criterion = CombinedLoss(["focal", "dice", "contour", "flag_bce"],
                             [0.4,    0.3,   0.1,      0.2])             # Sec 6
    opt    = build_optimizer(model)
    sched  = build_scheduler(opt, len(loader_tr))
    scaler = GradScaler()
    best   = -1

    # 4) training epochs
    for ep in range(CFG["epochs"]):
        model.train()
        for x, y, meta in loader_tr:
            x, y, meta = x.to(CFG["device"]), y.to(CFG["device"]), meta.to(CFG["device"])
            x, y = apply_sat_aug(x, y)                                   # Sec 3

            with autocast():
                feats   = backbone(x, meta)
                mask_lp, flag_lp = head(feats)
                loss = criterion(mask_lp, flag_lp, y)

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()

        # ----- validation -----
        val_score = evaluate(model, loader_va)                           # Sec 7
        print(f"ep{ep:02d}  val {val_score:.4f}")

        if val_score > best:
            best = val_score
            torch.save(model.state_dict(), "sat_best.pth")

    # 5) sweep optimal threshold for Dice / IoU blends
    sweep_threshold(model, loader_va, metric_name=CFG["metric_name"])    # Sec 7
    print("Training done. Best val =", best, "  Model saved to sat_best.pth")

# Example run:
# train_sat("/path/to/satellite_dataset")
# Section 9 · Inference + Test-Time Augmentation (S-9)
# ----------------------------------------------------
# One entry point ―  run_inference(test_root)
#   • Loads sat_best.pth + best_thr.json
#   • Optional horizontal flip TTA
#   • Writes masks to test_root/preds/*.pt   1-byte per pixel (uint8)
#
# Adjust CFG["use_tta"] to False if you need speed.

import json, torch, torch.nn.functional as F
from pathlib import Path

# -------- configuration flag --------
CFG["use_tta"] = True      # set False to disable flip-ensemble

# -------- rebuild model exactly as training --------
def load_model():
    meta_mod  = build_meta_block(512, mode="film", cond_dim=3)
    backbone  = UNetRes(in_ch=CFG["bands"],
                        dropout_p=0.1,
                        use_attn=True,
                        meta_block=meta_mod)
    head      = build_head(backbone.out_channels,
                           head_type="binary",
                           include_flag=True)
    model = nn.Sequential(backbone, head).to(CFG["device"])
    model.load_state_dict(torch.load("sat_best.pth", map_location=CFG["device"]))
    model.eval()
    return model

# -------- helper: flip TTA --------
def _forward_tta(model, x, meta):
    m1, f1 = model(x, meta)
    if not CFG["use_tta"]: return m1, f1
    x_flip = torch.flip(x, [-1])
    m2, f2 = model(x_flip, meta)
    m2 = torch.flip(m2, [-1])          # unflip
    m = (m1 + m2) / 2
    f = (f1 + f2) / 2
    return m, f

# -------- load threshold --------
try:
    _THR = json.load(open("best_thr.json"))["thr"]
except FileNotFoundError:
    _THR = CFG["thr"]

# -------- main inference routine --------
@torch.no_grad()
def run_inference(root):
    test_loader = make_loader(root, "test")      # uses collate_fn Section 2
    model = load_model()

    out_dir = Path(root, "preds"); out_dir.mkdir(exist_ok=True)
    for idx, (x, _, meta) in enumerate(test_loader):
        x, meta = x.to(CFG["device"]), meta.to(CFG["device"])
        m_logit, _ = _forward_tta(model, x, meta)
        masks = (torch.sigmoid(m_logit) > _THR).byte().cpu()   # uint8 0/1
        for i, mask in enumerate(masks):
            # save one file per sample   e.g. preds/idx_00012.pt
            torch.save(mask.squeeze(0), out_dir / f"idx_{idx*CFG['batch']+i:05d}.pt")
    print("Inference done. Masks saved to", out_dir)
# Section 10 · Few-Shot Adapt Helper  (S-10)
# -----------------------------------------------------------
# Quickly fine-tune sat_best.pth on a tiny organiser-supplied
# adaptation set (e.g. 10–50 images) and save adapted.pth.
#
# Key knobs
#   • freeze_encoder : True → only decoder + head learn
#   • epochs         : default 5  (fast)
#   • lr             : default 3e-4 (lower than full training)
#
# Usage
#   adapt_fewshot(adapt_root="adat_set",
#                 base_ckpt="sat_best.pth",
#                 out_ckpt="adapted.pth",
#                 freeze_encoder=True)

def adapt_fewshot(adapt_root,
                  base_ckpt="sat_best.pth",
                  out_ckpt="adapted.pth",
                  freeze_encoder=True,
                  epochs=5,
                  lr=3e-4):

    # ---------- loaders (reuse Section 2 make_loader) ----------
    tr = make_loader(adapt_root, "train")
    va = make_loader(adapt_root, "val")

    # ---------- rebuild model & load base weights -------------
    meta_mod = build_meta_block(512, mode="film", cond_dim=3)
    backbone = UNetRes(in_ch=CFG["bands"],
                       dropout_p=0.1,
                       use_attn=True,
                       meta_block=meta_mod)
    head     = build_head(backbone.out_channels,
                          head_type="binary",
                          include_flag=True)
    model = nn.Sequential(backbone, head).to(CFG["device"])
    model.load_state_dict(torch.load(base_ckpt, map_location=CFG["device"]))

    # freeze encoder if requested
    if freeze_encoder:
        for n, p in model.named_parameters():
            if "down" in n or "stem" in n or "mid" in n:
                p.requires_grad_(False)

    # ---------- optimiser / sched ----------
    opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=len(tr)*epochs)
    scaler, criterion = GradScaler(), CombinedLoss(
        ["focal","dice","flag_bce"], [0.5,0.3,0.2])

    best = -1
    for ep in range(epochs):
        model.train()
        for x, y, meta in tr:
            x,y,meta = x.to(CFG["device"]),y.to(CFG["device"]),meta.to(CFG["device"])
            x,y = apply_sat_aug(x,y, use_mixup=False, use_cutmix=False)  # light aug
            with autocast():
                feats = backbone(x, meta)
                m_log, f_log = head(feats)
                loss = criterion(m_log, f_log, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()

        # val metric
        val = evaluate(model, va)
        print(f"[adapt] ep{ep}  val {val:.4f}")
        if val > best:
            best = val; torch.save(model.state_dict(), out_ckpt)

    print("Few-shot adaptation done ➜", out_ckpt, "  best val =", best)

</code></pre>

</details>
<details>
<summary>Contributors (click to expand)</summary>

<pre><code class="language-python">
# ================================================================
# SECTION 1 · Text-Only Baseline (organiser MiniLM-L6 + fine-tune)
# ---------------------------------------------------------------
#  • load_icon_db()            → {id: description}
#  • encode_choices()          caches choice vectors
#  • guess_words(hints, opts)  returns top-10 prediction list
#  • fine_tune_20()            one-pass cosine-loss fine-tune on 20 val rounds
# ================================================================

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, util
import torch, json, math, random
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMB_DIM    = 384
TOP_K      = 10

# ---------- 0.  icon DB ----------
def load_icon_db(path="icon_descriptions.json"):
    return json.loads(Path(path).read_text())

ICON_DB = load_icon_db()            # {id: "A red apple …"}

# ---------- 1.  encoder + cache ----------
text_encoder = SentenceTransformer(BASE_MODEL, device=device)
_choice_cache = {}
def encode_choices(choices):
    miss = [c for c in choices if c not in _choice_cache]
    if miss:
        vecs = text_encoder.encode([f"A {x}" for x in miss],
                                   convert_to_tensor=True, show_progress_bar=False)
        for k,v in zip(miss, vecs):
            _choice_cache[k] = v / v.norm()
    return torch.stack([_choice_cache[c] for c in choices]).to(device)   # (N,384)

# ---------- 2.  hint prompt builder ----------
def hints_to_sentence(hints):
    # preserves order; works for 1–5 hints
    return " -> ".join([ICON_DB[h]['description'].lower() for h in hints])

# ---------- 3.  main guesser ----------
def guess_words(hints: list[int], choices: list[str]) -> list[str]:
    q   = hints_to_sentence(hints)
    qv  = text_encoder.encode(q, convert_to_tensor=True).to(device)
    qv  = qv / qv.norm()
    cv  = encode_choices(choices)                     # (N,384)
    sims= (qv @ cv.T).cpu()                           # (N,)
    top = sims.topk(TOP_K).indices
    return [choices[i] for i in top]

# ---------- 4.  optional fine-tune on 20 validation rounds ----------
def fine_tune_20(val_path="takehome_validation.json"):
    data  = json.loads(Path(val_path).read_text())
    rand  = random.Random(42)
    train = []
    for row in data:
        hints = [h for h in row['hints'] if h in ICON_DB]
        sent  = hints_to_sentence(hints)
        pos   = row['label']
        neg   = rand.choice([c for c in row['options'] if c != pos])
        train.append(InputExample(texts=[sent, f"A {pos}"], label=1.0))
        train.append(InputExample(texts=[sent, f"A {neg}"], label=0.0))
    ds   = SentencesDataset(train, text_encoder)
    loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=8)
    loss   = losses.CosineSimilarityLoss(text_encoder)
    text_encoder.fit(train_objectives=[(loader, loss)],
                     epochs=1, warmup_steps=10)
    print("⇒ Mini fine-tune done.")
# ================================================================
# SECTION 2 · CLIP Fusion (icons + descriptions) — Weather-team V2
# ---------------------------------------------------------------
#  • build_dataloaders()    loads 64×64 icon PNG + description
#  • train_clip_contrast() fine-tunes ViT-B/32 for 30 epochs
#  • clip_guess(hints, opts, α=0.5)  ranks by α·image+ (1-α)·text
# ================================================================

import os, torch, torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # mainland mirror
device   = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_name= 'openai/clip-vit-base-patch16'
clipM    = CLIPModel.from_pretrained(clip_name).to(device)
proc     = CLIPProcessor.from_pretrained(clip_name)

# ---------- 1.  dataset ----------
class IconSet(torch.utils.data.Dataset):
    def __init__(self, icon_db):
        self.ids  = sorted(icon_db)
        self.desc = [f"an icon showing {icon_db[i]['description'].replace('\n',' and ')}"
                     for i in self.ids]
        self.imgs = [icon_db[i]['icons'] for i in self.ids]  # PIL 64×64
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.imgs[i], self.desc[i]

def build_dataloaders(batch=32):
    ds = IconSet(ICON_DB)
    def collate(batch):
        imgs, txts = zip(*batch)
        enc = proc(images=list(imgs), text=list(txts), return_tensors='pt',
                   padding=True, truncation=True)
        return {k: v.to(device) for k,v in enc.items()}
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                       collate_fn=collate)

# ---------- 2.  contrastive fine-tune ----------
def train_clip_contrast(epochs=30, lr=5e-5):
    clipM.train(); opt = torch.optim.AdamW(clipM.parameters(), lr=lr)
    loader = build_dataloaders()
    for ep in range(epochs):
        tot = 0
        for batch in loader:
            opt.zero_grad()
            out = clipM(**batch)
            ie, te = F.normalize(out.image_embeds, p=2, dim=1), \
                     F.normalize(out.text_embeds,  p=2, dim=1)
            sim   = (ie @ te.T) * clipM.logit_scale.exp()
            tgt   = torch.arange(sim.size(0), device=device)
            loss  = (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt)) / 2
            loss.backward(); opt.step()
            tot += loss.item()
        if ep % 5 == 0: print(f"ep{ep:02d}  loss {tot/len(loader):.4f}")
    clipM.eval(); clipM.save_pretrained("./clip_ft"); proc.save_pretrained("./clip_ft")

# ---------- 3.  retrieval helper ----------
def clip_guess(hints, choices, alpha=0.5):
    # encode hints → imgs + desc
    imgs = [ICON_DB[h]['icons'] for h in hints]
    desc = [f"an icon showing {ICON_DB[h]['description'].replace('\n',' and ')}"
            for h in hints]
    enc_h = proc(images=imgs, text=desc, return_tensors='pt',
                 padding=True, truncation=True).to(device)
    enc_c = proc(text=[f"a {c}" for c in choices], return_tensors='pt',
                 padding=True, truncation=True).to(device)

    with torch.no_grad():
        ih = F.normalize(clipM.get_image_features(**enc_h), p=2, dim=1)
        th = F.normalize(clipM.get_text_features(**enc_h),  p=2, dim=1)
        tc = F.normalize(clipM.get_text_features(**enc_c),  p=2, dim=1)
        sim = alpha * (ih @ tc.T) + (1-alpha) * (th @ tc.T)
        score = sim.sum(0)
        top = score.topk(10).indices.cpu()
    return [choices[i] for i in top]

</code></pre>

</details>
