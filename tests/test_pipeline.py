"""
Regression and correctness checks.

  - simulators and feature computation reproduce frozen reference outputs
    (tests/fixtures/reference_outputs.npz), so refactoring cannot silently
    change the data or the features;
  - masks act on every channel copy of a feature and are appended for the flow;
  - the mask distribution matches its specification;
  - the importance read-out is right on hand-computed and brute-force cases,
    and the toy's ground truth has its planted properties;
  - the support correction matches an analytic case.

Run from the repository root:  python tests/test_pipeline.py
(also collectable by pytest)

After an intentional change to the simulators or features, refresh the frozen
outputs with:  python tests/test_pipeline.py --regenerate
"""

import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config                                       # noqa: E402
from features import FeatureNet, PassThroughFeatureNet          # noqa: E402
from importance import importance_from_values                  # noqa: E402
from masked_npe import MaskedFeatureNet                        # noqa: E402
from toy import ToyConfig, ToyModel                            # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "reference_outputs.npz"
# Tolerance for float32 differences across CPUs / BLAS builds.
TOL = dict(rtol=1e-5, atol=1e-5)


# ─── Reference outputs ───────────────────────────────────────────────────────
# Each function rebuilds fixed inputs from seeds and runs the current code. The
# regression tests compare these against FIXTURE; --regenerate overwrites it.

def _jansen_rit_outputs():
    from jansen_rit import generate_sobol_parameters, simulate_jansen_rit

    cfg = Config()
    # Observation noise on, so the (optional) noise code stays covered.
    cfg.N_CHANNELS, cfg.USE_FORWARD_MODEL, cfg.N_CONTROL_NOISE, cfg.ADD_NOISE = 1, False, 1, True
    lo, hi, _ = cfg.get_effective_prior_bounds()
    theta = generate_sobol_parameters(4, lo, hi, seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    x = simulate_jansen_rit(theta, cfg, use_gpu=False)
    return {"jr_theta": theta.numpy(), "jr_sim": x.numpy()}


def _toy_data():
    model = ToyModel(ToyConfig(n_channels=2, n_control=1))
    theta = model.sample_theta(256, generator=torch.Generator().manual_seed(1))
    return model, model.simulate(theta, generator=torch.Generator().manual_seed(2))


def _toy_outputs():
    model, x = _toy_data()
    b = PassThroughFeatureNet(base_feature_names=model.feature_names, n_channels=2, n_control=1,
                              excluded_features=["f5_null"])
    b.compute_normalization_stats(x)
    with torch.no_grad():
        return {"toy_sim": x.numpy(), "features_toy_excl": b(x).numpy()}


def _jr_feature_outputs():
    g = torch.Generator().manual_seed(0)
    n, ch, T, n_ctrl = 64, 5, 750, 2
    # Mixed scales, offsets and a constant channel, to exercise every guard.
    sig = torch.randn(n, ch, T, generator=g) * torch.logspace(-9, 2, n).view(n, 1, 1) + 3.0
    sig[0, 1] = 1.0
    ctrl = torch.randn(n, n_ctrl, 1, generator=g).expand(-1, -1, T)
    x = torch.cat([sig, ctrl], dim=1)
    out = {}
    for tag, excluded in (("jr_all", []), ("jr_excl", ["kurtosis", "noise_1"])):
        b = FeatureNet(T, fs=250, excluded_features=excluded, n_channels=ch, n_control=n_ctrl)
        b.compute_normalization_stats(x)
        with torch.no_grad():
            out[f"features_{tag}"] = b(x).numpy()
    return out


def reference_outputs():
    return {**_jansen_rit_outputs(), **_toy_outputs(), **_jr_feature_outputs()}


def regenerate():
    """Overwrite FIXTURE with the current code's outputs. Only do this after an
    intentional change to the simulators or features."""
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **reference_outputs())
    print(f"wrote {FIXTURE}")


def _assert_match(outputs):
    reference = np.load(FIXTURE)
    for key, value in outputs.items():
        assert np.allclose(value, reference[key], **TOL), f"{key} differs from {FIXTURE.name}"


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_jansen_rit_simulator_matches_reference():
    _assert_match(_jansen_rit_outputs())


def test_jansen_rit_training_data_reproducible_from_seed():
    from problems import JansenRitProblem

    cfg = Config()
    # Noise on: it is drawn from numpy, the RNG that is easiest to leave unseeded.
    cfg.NUM_SIMULATIONS, cfg.N_CHANNELS, cfg.USE_FORWARD_MODEL, cfg.ADD_NOISE = 8, 1, False, True
    problem = JansenRitProblem(cfg)
    theta_a, x_a = problem.simulate_train(seed=3)
    np.random.rand(5)                  # disturb the global RNG state in between
    theta_b, x_b = problem.simulate_train(seed=3)
    _, x_c = problem.simulate_train(seed=4)
    assert torch.equal(theta_a, theta_b) and torch.equal(x_a, x_b)
    assert not torch.equal(x_a, x_c)


def test_default_jansen_rit_setup():
    """One channel, no observation noise, no forward model, signal_mean excluded."""
    from problems import JansenRitProblem

    cfg = Config()
    assert "signal_mean" in cfg.EXCLUDED_FEATURES
    assert (cfg.N_CHANNELS, cfg.ADD_NOISE, cfg.USE_FORWARD_MODEL) == (1, False, False)
    cfg.NUM_SIMULATIONS = 8
    problem = JansenRitProblem(cfg)
    _, x = problem.simulate_train(seed=0)
    net = problem.feature_net()
    assert x.shape == (8, cfg.SIGNAL_LENGTH)
    assert "signal_mean" not in net.groups and len(net.groups) == 7 and net.n_features == 7


def test_toy_problem_builds_with_default_config():
    """Jansen-Rit-only exclusions (signal_mean) must not break the toy; typos must."""
    from problems import ToyProblem

    cfg = Config()
    cfg.N_CONTROL_NOISE = 1
    net = ToyProblem(cfg).feature_net()
    assert len(net.groups) == 9                      # 8 toy features + 1 control
    cfg.EXCLUDED_FEATURES = cfg.EXCLUDED_FEATURES + ["not_a_feature"]
    try:
        ToyProblem(cfg).feature_net()
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown exclusion should raise")


def test_single_feature_net():
    """The single-feature references keep exactly one group."""
    from problems import JansenRitProblem

    cfg = Config()
    problem = JansenRitProblem(cfg)
    groups = problem.feature_net().groups
    net = problem.feature_net(excluded=[h for h in groups if h != "kurtosis"])
    assert net.groups == ["kurtosis"] and net.n_features == 1


def test_results_folder_replaces_seeds_and_refuses_other_settings():
    import tempfile
    import pandas as pd
    import results_log as rlog

    with tempfile.TemporaryDirectory() as tmp:
        rlog.RESULTS_DIR, saved = Path(tmp), rlog.RESULTS_DIR
        try:
            cfg = Config()
            rlog.write_seed("exp", cfg, 0, {"seeds": [{"x": 1.0}]})
            rlog.write_seed("exp", cfg, 1, {"seeds": [{"x": 2.0}]})
            rlog.write_seed("exp", cfg, 0, {"seeds": [{"x": 3.0}]})          # rerun of seed 0
            seeds = pd.read_csv(Path(tmp) / "exp" / "seeds.csv")
            assert seeds.set_index("seed")["x"].to_dict() == {0: 3.0, 1: 2.0}
            cfg.SEED = 7                                                    # the seed is not a setting
            rlog.write_seed("exp", cfg, 2, {"seeds": [{"x": 4.0}]})
            cfg.NUM_SIMULATIONS += 1
            try:
                rlog.write_seed("exp", cfg, 3, {"seeds": [{"x": 5.0}]})
            except ValueError as err:
                assert "NUM_SIMULATIONS" in str(err)
            else:
                raise AssertionError("different settings must not share a folder")
        finally:
            rlog.RESULTS_DIR = saved


def test_toy_simulator_and_features_match_reference():
    _assert_match(_toy_outputs())


def test_jr_features_match_reference():
    _assert_match(_jr_feature_outputs())


def _toy_wrapper(p_full=0.0, p_loo=0.0):
    model = ToyModel(ToyConfig(n_channels=2, n_control=1))
    x = model.simulate(model.sample_theta(32))
    base = PassThroughFeatureNet(base_feature_names=model.feature_names, n_channels=2, n_control=1)
    base.compute_normalization_stats(x)
    return MaskedFeatureNet(base, p_full=p_full, p_loo=p_loo), x


def test_mask_applies_to_all_channel_copies_and_is_appended():
    w, x = _toy_wrapper()
    G, F = w.n_groups, w.base.n_features
    w.mask_mode = "fixed"
    keep = [g for g in w.groups if g != "f2_strong_t2"]
    w.fixed_mask.copy_(torch.tensor([1.0 if g in keep else 0.0 for g in w.groups]))
    with torch.no_grad():
        out = w(x)
    assert out.shape[1] == F + G
    cols = [i for i in range(F) if w.base._base_feature_of(i) == "f2_strong_t2"]
    assert len(cols) == 2 and torch.all(out[:, cols] == 0)
    assert torch.equal(out[:, F:], w.fixed_mask.expand(len(x), -1))
    w.mask_mode = "full"
    with torch.no_grad():
        assert torch.equal(w(x)[:, :F], w.base(x))


def test_mask_distribution():
    torch.manual_seed(0)
    w, _ = _toy_wrapper()
    m = w.sample_masks(200_000, "cpu")
    G = w.n_groups
    sizes = m.sum(1)
    for k in range(G + 1):     # size-uniform scheme
        assert abs((sizes == k).float().mean().item() - 1 / (G + 1)) < 0.01
    w2, _ = _toy_wrapper(p_full=0.2, p_loo=0.3)
    s2 = w2.sample_masks(200_000, "cpu").sum(1)
    base = 0.5 / (G + 1)
    assert abs((s2 == G).float().mean().item() - (0.2 + base)) < 0.01
    assert abs((s2 == G - 1).float().mean().item() - (0.3 + base)) < 0.01


def test_validation_masks_fixed_per_point_with_training_distribution():
    model = ToyModel(ToyConfig(n_channels=2, n_control=1))
    x = model.simulate(model.sample_theta(50_000, generator=torch.Generator().manual_seed(5)),
                       generator=torch.Generator().manual_seed(6))
    base = PassThroughFeatureNet(base_feature_names=model.feature_names, n_channels=2, n_control=1)
    base.compute_normalization_stats(x)
    w = MaskedFeatureNet(base)
    G, F = w.n_groups, base.n_features
    perm = torch.randperm(len(x))
    with torch.no_grad():
        w.eval()                       # sbi's validation pass
        m = w(x)[:, F:]
        assert torch.equal(m, w(x)[:, F:])                       # same every epoch
        assert torch.equal(m[perm], w(x[perm])[:, F:])           # independent of batch order
        w.train()                      # training: fresh draws
        assert not torch.equal(w(x[:1000])[:, F:], w(x[:1000])[:, F:])
    sizes = m.sum(1)
    for k in range(G + 1):             # same size-uniform distribution as training
        assert abs((sizes == k).float().mean().item() - 1 / (G + 1)) < 0.01
    assert torch.all((m.mean(0) - 0.5).abs() < 0.01)             # each group kept half the time


def test_importance_math_on_hand_computed_example():
    # v(empty)=0, v({0})=1, v({1})=2, v({0,1})=4, worked out by hand.
    v = {frozenset(): 0.0, frozenset([0]): 1.0, frozenset([1]): 2.0, frozenset([0, 1]): 4.0}
    est = importance_from_values(v, 2)
    assert np.allclose(est["unique"], [2.0, 3.0])
    assert np.allclose(est["marginal"], [1.0, 2.0])
    assert np.allclose(est["shapley"], [1.5, 2.5])


def test_shapley_matches_average_over_orderings():
    """Shapley = j's average gain over every order of adding the features."""
    n = 4
    rng = np.random.default_rng(0)
    v = {frozenset(c): float(rng.normal()) for k in range(n + 1) for c in combinations(range(n), k)}
    brute = np.zeros(n)
    orders = list(permutations(range(n)))
    for order in orders:
        seen = frozenset()
        for j in order:
            brute[j] += v[seen | {j}] - v[seen]
            seen = seen | {j}
    assert np.allclose(importance_from_values(v, n)["shapley"], brute / len(orders))


def test_toy_exact_log_posterior():
    model = ToyModel(ToyConfig(n_channels=2, n_control=1))
    theta = model.sample_theta(200_000, generator=torch.Generator().manual_seed(7))
    x = model.simulate(theta, generator=torch.Generator().manual_seed(8))[:, :, 0].numpy()
    theta = theta.numpy()
    prior = model.log_posterior(theta, x, [])
    sd = np.asarray(model.config.theta_sd)
    expected_prior = (-0.5 * ((theta / sd) ** 2).sum(1) - np.log(sd).sum()
                      - 0.5 * len(sd) * np.log(2 * np.pi))
    assert np.allclose(prior, expected_prior)                    # empty subset = prior
    # Averaged over the joint distribution, log p(theta|x_S) - log p(theta) = I(theta; x_S).
    for subset in ([0], [5], [0, 5], [2, 3], list(range(8))):
        gain = model.log_posterior(theta, x, subset) - prior
        se = gain.std() / np.sqrt(len(gain))
        assert abs(gain.mean() - model.info(subset)) <= 4 * se + 1e-9, subset


def test_toy_ground_truth_properties():
    model = ToyModel(ToyConfig())
    gt = model.ground_truth("info")
    names = model.feature_names
    n = model.n_base_features
    # Shapley values add up to the total information.
    assert np.isclose(gt["shapley"].sum(), model.info(range(n)) - model.info([]))
    # The null carries nothing under any estimand; the suppressor is useless
    # alone but valuable given f1; the duplicates are redundant with each other.
    null, supp, dup = names.index("f5_null"), names.index("f6_suppressor"), names.index("f3_dup_a")
    assert all(abs(gt[e][null]) < 1e-9 for e in ("unique", "marginal", "shapley"))
    assert abs(gt["marginal"][supp]) < 1e-9 and gt["unique"][supp] > 1e-2
    assert gt["unique"][dup] < 0.2 * gt["marginal"][dup]


def test_support_mass_matches_analytic():
    """log Z for a standard normal on [-1, 1]: Z = 0.6827; unbounded -> exactly 0."""
    from evaluation import log_support_mass

    class StdNormal:
        def sample(self, shape, condition):
            return torch.randn(*shape, condition.shape[0], 1)

    torch.manual_seed(0)
    x = torch.zeros(50, 3)
    bounds = (torch.tensor([-1.0]), torch.tensor([1.0]))
    z = np.exp(log_support_mass(StdNormal(), x, bounds, n_samples=4000))
    assert abs(z.mean() - 0.6827) < 0.01, z.mean()
    assert np.all(log_support_mass(StdNormal(), x, None, n_samples=4000) == 0)


if __name__ == "__main__":
    if "--regenerate" in sys.argv[1:]:
        regenerate()
        sys.exit(0)
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:          # report every failure, not just the first
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
