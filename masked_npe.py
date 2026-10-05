"""
Masked NPE: one posterior estimator that conditions on ANY subset of features.

    x -> FeatureNet -> z [batch, F] -> [ z * m_cols , m ] -> NSF flow -> q(theta | x_S)

m has one 0/1 entry per *group*: a base feature (all channel copies move
together, as in ablation, which removes a feature from every channel) or a
pass-through noise control. Appending m lets the flow tell a hidden feature
from an observed one sitting at its mean (z = 0).

Training draws a fresh random mask per sample, so one network learns
q(theta | x_S) for every S; at evaluation a fixed mask selects the subset.
Removal therefore stays inside the training distribution (unlike permutation
or occlusion) and needs no retraining (unlike retrain ablation).

Mask modes (MaskedFeatureNet.mask_mode):
    "random"  random masks from the training distribution, for training AND
              sbi's validation pass, so early stopping tracks the objective
              being optimized. In train() mode every row gets a fresh mask;
              in eval() mode (sbi's validation pass) each row's mask is a
              fixed function of its features, so the validation loss does not
              change from epoch to epoch just because the masks were redrawn.
    "fixed"   the mask set by set_fixed_mask(), for every row
    "full"    all groups present: the ordinary full-feature posterior

sbi holds deep copies of the embedding net (caller's, trained estimator's,
posterior's). A mode set on the caller's handle never reaches the copy that is
evaluated, so always use set_mask_mode()/set_fixed_mask() on the posterior,
which update every MaskedFeatureNet it contains and fail loudly if none.
"""

from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn

from features import FeatureNet


class MaskedFeatureNet(nn.Module):
    """
    Wraps a FeatureNet and applies group masks to its z-scored output.

    Mask distribution during training (per sample):
        with prob p_full        -> all groups
        else with prob p_loo    -> all groups but one (uniformly chosen)
        otherwise               -> size k ~ Uniform{0..G}, uniform subset of size k

    p_full = p_loo = 0 is the plain size-uniform scheme, under which any one
    leave-one-out subset gets only ~1/(G(G+1)) of the training masks.
    """

    MODES = ("random", "fixed", "full")

    def __init__(self, base: FeatureNet, p_full: float = 0.0, p_loo: float = 0.0):
        super().__init__()
        for name, p in (("p_full", p_full), ("p_loo", p_loo)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {p}")
        if p_full + p_loo > 1.0:
            raise ValueError(f"p_full + p_loo must be <= 1, got {p_full + p_loo}")
        self.base = base
        self.groups: List[str] = base.groups
        self.n_groups = len(self.groups)
        col_group = [self.groups.index(base._base_feature_of(i)) for i in range(base.n_features)]
        self.register_buffer("col_group", torch.tensor(col_group, dtype=torch.long))
        self.register_buffer("fixed_mask", torch.ones(self.n_groups))
        self.p_full = p_full
        self.p_loo = p_loo
        self.mask_mode = "random"
        # Fixed random projection used to hash a row's features into the
        # uniforms that define its validation mask (see feature_masks).
        proj = torch.randn(base.n_features, self.n_groups + 2,
                           generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        self.register_buffer("hash_proj", proj * 1e3)

    def _masks(self, k: torch.Tensor, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Masks from a drawn size k [batch, 1], mixture uniform u [batch, 1]
        and ranking uniforms r [batch, G] (mixture in the class docstring)."""
        G = self.n_groups
        k = torch.where(u < self.p_full, torch.full_like(k, G), k)
        loo = (u >= self.p_full) & (u < self.p_full + self.p_loo)
        k = torch.where(loo, torch.full_like(k, G - 1), k)
        # rank of each group under a random permutation; keep the k lowest
        ranks = r.argsort(dim=1).argsort(dim=1)
        return (ranks < k).float()

    def sample_masks(self, batch: int, device) -> torch.Tensor:
        """[batch, G] fresh random training masks."""
        G = self.n_groups
        k = torch.randint(0, G + 1, (batch, 1), device=device)
        u = torch.rand(batch, 1, device=device)
        return self._masks(k, u, torch.rand(batch, G, device=device))

    def feature_masks(self, z: torch.Tensor) -> torch.Tensor:
        """[batch, G] masks with the training distribution, each a fixed
        function of its row's features: the same point always gets the same
        mask, whatever the batch or its order."""
        h = torch.sin(z.double() @ self.hash_proj) * 43758.5453
        h = (h - torch.floor(h)).float()                   # uniforms in [0, 1)
        G = self.n_groups
        k = (h[:, :1] * (G + 1)).long().clamp(max=G)
        return self._masks(k, h[:, 1:2], h[:, 2:])

    def current_masks(self, z: torch.Tensor) -> torch.Tensor:
        batch, device = z.shape[0], z.device
        if self.mask_mode == "random":
            return self.sample_masks(batch, device) if self.training else self.feature_masks(z)
        if self.mask_mode == "fixed":
            return self.fixed_mask.to(device).expand(batch, -1)
        return torch.ones(batch, self.n_groups, device=device)

    def forward(self, x):
        z = self.base(x)                                   # [batch, F]
        m = self.current_masks(z)                          # [batch, G]
        return torch.cat([z * m[:, self.col_group], m], dim=1)


# ─── Controlling the copies sbi actually evaluates ───────────────────────────

def masked_nets(obj) -> List[MaskedFeatureNet]:
    """Every MaskedFeatureNet inside a posterior, estimator or module."""
    found, seen = [], set()
    for attr in ("posterior_estimator", "net", "_neural_net", None):
        target = obj if attr is None else getattr(obj, attr, None)
        if target is None or not hasattr(target, "modules"):
            continue
        for m in target.modules():
            if isinstance(m, MaskedFeatureNet) and id(m) not in seen:
                seen.add(id(m))
                found.append(m)
    if not found:
        raise RuntimeError("No MaskedFeatureNet found; a mask would be silently ignored.")
    return found


def set_mask_mode(obj, mode: str) -> None:
    if mode not in MaskedFeatureNet.MODES:
        raise ValueError(f"mode must be one of {MaskedFeatureNet.MODES}, got '{mode}'")
    for net in masked_nets(obj):
        net.mask_mode = mode


def set_fixed_mask(obj, present: Iterable[str]) -> None:
    """Condition on exactly the named groups; all others hidden."""
    nets = masked_nets(obj)
    present = set(present)
    unknown = present - set(nets[0].groups)
    if unknown:
        raise ValueError(f"Unknown groups {sorted(unknown)}; have {nets[0].groups}")
    for net in nets:
        mask = torch.tensor([1.0 if g in present else 0.0 for g in net.groups],
                            device=net.fixed_mask.device)
        net.fixed_mask.copy_(mask)
        net.mask_mode = "fixed"


def resolve_estimator(posterior):
    """The density estimator a posterior wraps, for direct batched log_prob."""
    for attr in ("posterior_estimator", "net", "_neural_net"):
        est = getattr(posterior, attr, None)
        if est is not None and hasattr(est, "log_prob"):
            return est
    raise RuntimeError("Could not locate a density estimator with .log_prob on the posterior.")


# ─── Training ────────────────────────────────────────────────────────────────

class _NoTensorBoard:
    """Stand-in for sbi's SummaryWriter: sbi only calls add_scalar/flush, and its
    default writer leaves one TensorBoard folder per training in ./sbi-logs.
    The per-epoch losses are kept instead via training_curve() -> training_curves.csv."""

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass


def training_curve(summary) -> dict:
    """Per-epoch training/validation loss from sbi's training summary."""
    return {"training_loss": [float(v) for v in summary.get("training_loss", [])],
            "validation_loss": [float(v) for v in summary.get("validation_loss", [])]}


def train_npe(embedding_net: nn.Module, prior, theta_train, x_train, cfg, seed: int,
              device, verbose: bool = False):
    """
    Train an NSF NPE with the estimator settings in cfg. Shared by the masked
    NPE and the references, so every model in a comparison is trained identically.

    sbi's own x z-scoring is off: the feature net z-scores with stats from the
    raw training signals. sbi's Standardize layer would sit in front of it, so
    during training the net would see standardized input normalized with
    raw-input stats, which no longer matches the input it gets at evaluation.

    Returns (posterior, summary).
    """
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    inference = SNPE(
        prior=prior,
        density_estimator=posterior_nn(
            model="nsf", embedding_net=embedding_net, z_score_x="none",
            hidden_features=cfg.HIDDEN_FEATURES, num_transforms=cfg.NUM_TRANSFORMS),
        device=str(device),
        summary_writer=_NoTensorBoard(),
    )
    estimator = inference.append_simulations(theta_train, x_train).train(
        learning_rate=cfg.LEARNING_RATE,
        max_num_epochs=cfg.MAX_EPOCHS,
        stop_after_epochs=cfg.STOP_AFTER_EPOCHS,
        training_batch_size=cfg.BATCH_SIZE,
        validation_fraction=cfg.VALIDATION_FRACTION,
        show_train_summary=verbose,
    )
    return inference.build_posterior(estimator), inference._summary


def train_masked_npe(base: FeatureNet, prior, theta_train, x_train, cfg, seed: int,
                     device, verbose: bool = False):
    """Masked NPE with cfg.MASK_P_FULL / MASK_P_LOO. Returns (posterior, wrapper, summary)."""
    base.compute_normalization_stats(x_train)
    wrapper = MaskedFeatureNet(base, p_full=cfg.MASK_P_FULL, p_loo=cfg.MASK_P_LOO)
    wrapper.mask_mode = "random"
    posterior, summary = train_npe(wrapper, prior, theta_train, x_train, cfg, seed, device, verbose)
    set_mask_mode(posterior, "full")
    wrapper.mask_mode = "full"
    return posterior, wrapper, summary


def epochs_trained(summary) -> int:
    e = summary.get("epochs_trained", [0])
    return int(e[-1] if isinstance(e, (list, np.ndarray)) and len(e) else e)
