"""
Masked-NPE feature importance, on the toy or on Jansen-Rit.

One code path for both simulators (problems.py hides the differences):

  1. simulate training data (seed) and a fixed eval set (cfg.EVAL_SEED)
  2. train ONE masked NPE
  3. R^2 with all features and with each left out (posterior sampling)
  4. log q for every feature subset (batched) -> unique / marginal / Shapley
  5. compare with references:
       toy -- exact posterior log-density on the same eval points, for every
              subset and every estimand (always)
       --reference full      ordinary full-feature NPE (cost of masking)
       --reference ablation  + dedicated retrain per removed feature
       --reference marginal  dedicated NPE per single feature, for marginal
  6. write the seed's rows to results/<out>/ (results_log.py), replacing any
     earlier rows for that seed

Usage:
    python run.py --simulator toy --seeds 0 1 2 --reference ablation
    python run.py --simulator jr  --seeds 0 --reference ablation
    python run.py --simulator jr  --seeds 0 --p_full 0.2 --p_loo 0.2 --out jansen_rit_pfull
    python run.py --simulator toy --smoke
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

import results_log as rlog
from config import Config
from evaluation import compare_to_reference, r2_scores
from importance import (ESTIMANDS, importance_from_values, importance_table,
                        score_all_subsets, subset_table)
from masked_npe import epochs_trained, train_masked_npe, training_curve
from problems import DEVICE, make_problem


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulator", choices=["toy", "jr"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--num_simulations", type=int, default=None,
                    help="Default: 16384 for jr (Config), 8192 for toy")
    ap.add_argument("--n_control", type=int, default=0, help="Pure-noise null features")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Features removed from every model, added to Config.EXCLUDED_FEATURES")
    ap.add_argument("--p_full", type=float, default=None, help="Mask: extra prob of full set")
    ap.add_argument("--p_loo", type=float, default=None, help="Mask: prob of a leave-one-out subset")
    ap.add_argument("--reference", choices=["none", "full", "ablation", "marginal"], default="none")
    ap.add_argument("--r2", choices=["loo", "full", "none"], default=None)
    ap.add_argument("--n_eval_logprob", type=int, default=None)
    ap.add_argument("--n_channels", type=int, default=1, help="toy only")
    ap.add_argument("--dup_jitter", type=float, default=None, help="toy only: redundancy knob")
    ap.add_argument("--out", type=str, default=None,
                    help="Results folder under results/ (default: jansen_rit or toy, plus "
                         "'_marginal' for --reference marginal and '_smoke' for --smoke). "
                         "One folder holds one configuration.")
    ap.add_argument("--smoke", action="store_true", help="Tiny, fast wiring check")
    return ap.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    if args.num_simulations is not None:
        cfg.NUM_SIMULATIONS = args.num_simulations
    elif args.simulator == "toy":
        cfg.NUM_SIMULATIONS = 8192
    cfg.N_CONTROL_NOISE = args.n_control
    cfg.EXCLUDED_FEATURES = list(dict.fromkeys(cfg.EXCLUDED_FEATURES + list(args.exclude)))
    for attr, val in (("MASK_P_FULL", args.p_full), ("MASK_P_LOO", args.p_loo),
                      ("R2_SUBSETS", args.r2), ("N_EVAL_LOGPROB", args.n_eval_logprob)):
        if val is not None:
            setattr(cfg, attr, val)
    if args.smoke:
        cfg.NUM_SIMULATIONS, cfg.N_EVAL_SAMPLES, cfg.N_EVAL_LOGPROB = 1024, 30, 300
        cfg.MAX_EPOCHS, cfg.STOP_AFTER_EPOCHS = 5, 3
        cfg.HIDDEN_FEATURES, cfg.NUM_TRANSFORMS, cfg.N_POSTERIOR_SAMPLES = 32, 2, 200
    return cfg


SIMULATOR_NAMES = {"jr": "jansen_rit", "toy": "toy"}


def output_name(args) -> str:
    """results/ subfolder: --out, else e.g. jansen_rit, jansen_rit_marginal, toy_smoke."""
    if args.out:
        return args.out
    return (SIMULATOR_NAMES[args.simulator] + ("_marginal" if args.reference == "marginal" else "")
            + ("_smoke" if args.smoke else ""))


def recovery_row(model, left_out, r2, nrmse, cov90, full_r2, param_names):
    """One parameter_recovery.csv row; left_out is "" for the model with all features."""
    row = {"model": model, "left_out_feature": left_out,
           "r2_mean": float(np.mean(r2)), "r2_drop_mean": float(np.mean(full_r2) - np.mean(r2))}
    for i, p in enumerate(param_names):
        row.update({f"r2_{p}": float(r2[i]), f"r2_drop_{p}": float(full_r2[i] - r2[i]),
                    f"nrmse_{p}": float(nrmse[i]), f"coverage90_{p}": float(cov90[i])})
    return row


def add_agreement(run, prefix, estimate, reference):
    """Spearman / Pearson / MAE between two {feature: value} maps, as run columns."""
    for k, v in compare_to_reference(estimate, reference).items():
        if k != "n":
            run[f"{prefix}_{k}"] = v


def run_seed(problem, seed, args):
    cfg = problem.cfg
    cfg.SEED = seed
    params = problem.param_names
    print(f"\n=== {output_name(args)}  seed={seed}  sims={cfg.NUM_SIMULATIONS}"
          f"  p_full={cfg.MASK_P_FULL}  p_loo={cfg.MASK_P_LOO}")

    # 1. data
    t0 = time.time()
    theta_train, x_train = problem.simulate_train(seed)
    n_lp, n_r2 = cfg.N_EVAL_LOGPROB, cfg.N_EVAL_SAMPLES
    theta_eval, x_eval = problem.eval_set(max(n_lp, n_r2))
    th_lp, x_lp, th_r2, x_r2 = theta_eval[:n_lp], x_eval[:n_lp], theta_eval[:n_r2], x_eval[:n_r2]
    print(f"  data: {len(theta_train)} train, {len(theta_eval)} eval ({time.time() - t0:.0f}s)")

    # 2. masked NPE
    t0 = time.time()
    posterior, wrapper, summary = train_masked_npe(problem.feature_net(), problem.prior,
                                                   theta_train, x_train, cfg, seed, DEVICE)
    groups, t_train = wrapper.groups, time.time() - t0
    print(f"  masked NPE: {epochs_trained(summary)} epochs, {t_train:.0f}s; groups={groups}")

    # 3. R^2 by posterior sampling
    r2_rows, masked_full_r2 = [], None
    if cfg.R2_SUBSETS != "none":
        conds = [("", groups)]
        if cfg.R2_SUBSETS == "loo":
            conds += [(g, [h for h in groups if h != g]) for g in groups]
        for left_out, present in conds:
            r2, nrmse, cov = r2_scores(posterior, cfg, th_r2, x_r2, present)
            if not left_out:
                masked_full_r2 = r2
            r2_rows.append(recovery_row("masked_npe", left_out, r2, nrmse, cov, masked_full_r2, params))
        print(f"  R2 (masked): full={np.mean(masked_full_r2):.3f}" +
              ("  LOO drops: " + ", ".join(f"{r['left_out_feature']}={r['r2_drop_mean']:+.3f}"
                                           for r in r2_rows[1:]) if len(r2_rows) > 1 else ""))

    # 4. every subset by log-prob
    t0 = time.time()
    _, lp, log_z = score_all_subsets(posterior, th_lp, x_lp, verbose=False,
                                     bounds=problem.support_bounds,
                                     n_support_samples=cfg.N_SUPPORT_SAMPLES)
    imp = importance_table(groups, lp)
    sub = subset_table(groups, lp, prior_entropy=problem.prior_entropy,
                       log_z=log_z if problem.support_bounds is not None else None)
    print(f"  scored {len(lp)} subsets on {n_lp} points ({time.time() - t0:.0f}s)")

    G = len(groups)
    is_all, is_none = sub["n_features"] == G, sub["n_features"] == 0
    run = {"features": groups, "n_features": G,
           "masked_epochs": epochs_trained(summary), "masked_train_seconds": round(t_train, 1),
           "prior_entropy": problem.prior_entropy,
           "masked_logprob_all": float(sub.loc[is_all, "logprob"].iloc[0]),
           "masked_logprob_none": float(sub.loc[is_none, "logprob"].iloc[0])}
    run["masked_info_lower_bound_all"] = run["masked_logprob_all"] + problem.prior_entropy
    if "in_prior_mass" in sub:
        # Posterior mass inside the prior box: all features, leave-one-out sets, every subset.
        run.update(in_prior_mass_all=float(sub.loc[is_all, "in_prior_mass"].iloc[0]),
                   in_prior_mass_leave_one_out=float(sub.loc[sub["n_features"] == G - 1, "in_prior_mass"].mean()),
                   in_prior_mass_mean=float(sub["in_prior_mass"].mean()))
    if masked_full_r2 is not None:
        run["masked_r2_all"] = float(np.mean(masked_full_r2))
    masked = {e: dict(zip(imp[imp.estimand == e].feature, imp[imp.estimand == e].value)) for e in ESTIMANDS}
    imp_rows = [{"method": "masked_npe", **r} for r in imp.to_dict("records")]

    # 5a. exact truth (toy): the exact posterior scored on the SAME eval points,
    # so estimate and truth share their evaluation-set noise and differences
    # isolate the masked NPE's error. gap = v_exact - v estimates the KL from
    # the true posterior (>= 0 in expectation).
    exact = None
    if problem.exact is not None:
        v_exact = {s: float(problem.exact_logprob(th_lp, x_lp, [groups[i] for i in s]).mean()) for s in lp}
        members = [frozenset(i for i, c in enumerate(m) if c == "1") for m in sub["mask"]]
        sub["exact_info"] = [problem.exact_info([groups[i] for i in s]) for s in members]
        sub["exact_logprob"] = [v_exact[s] for s in members]
        sub["gap"] = sub["exact_logprob"] - sub["logprob"]
        run.update(exact_info_all=float(sub["exact_info"].max()),
                   exact_logprob_all=v_exact[frozenset(range(G))],
                   masked_gap_all=float(sub.loc[is_all, "gap"].iloc[0]),
                   masked_gap_none=float(sub.loc[is_none, "gap"].iloc[0]),
                   masked_gap_mean=float(sub["gap"].mean()))
        exact = {e: dict(zip(groups, vals.tolist()))
                 for e, vals in importance_from_values(v_exact, G).items()}
        for e in ESTIMANDS:
            imp_rows += [{"method": "exact", "feature": f, "estimand": e, "value": v}
                         for f, v in exact[e].items()]
            add_agreement(run, f"{e}_vs_exact", masked[e], exact[e])

    # 5b. references
    retrain = None
    if args.reference in ("full", "ablation"):
        from references import score, train_full_npe
        print("  reference: full-feature NPE")
        ref_post, ref_info = train_full_npe(problem, theta_train, x_train, seed, DEVICE)
        use_r2 = cfg.R2_SUBSETS != "none"
        full_scores = score(problem, ref_post, th_lp, x_lp, th_r2 if use_r2 else None, x_r2 if use_r2 else None)
        run.update(dedicated_logprob_all=full_scores["logprob"], dedicated_epochs=ref_info["epochs"],
                   dedicated_train_seconds=ref_info["train_time_s"])
        if "r2" in full_scores:
            run["dedicated_r2_all"] = float(np.mean(full_scores["r2"]))
            r2_rows.append(recovery_row("dedicated_npe", "", full_scores["r2"], full_scores["nrmse"],
                                        full_scores["coverage_90"], full_scores["r2"], params))
        if exact is not None:
            run["dedicated_gap_all"] = run["exact_logprob_all"] - full_scores["logprob"]
        del ref_post

        if args.reference == "ablation":
            from references import retrain_ablation
            print("  reference: retrain-ablation")
            retrain = retrain_ablation(problem, groups, theta_train, x_train, seed, DEVICE, full_scores,
                                       th_lp, x_lp, th_r2 if use_r2 else None, x_r2 if use_r2 else None)
            imp_rows += [{"method": "retrain_ablation", "feature": g, "estimand": "unique",
                          "value": r["logprob_drop"]} for g, r in retrain.items()]
            add_agreement(run, "unique_vs_retrain", masked["unique"],
                          {g: r["logprob_drop"] for g, r in retrain.items()})
            if use_r2:
                for g, r in retrain.items():
                    r2_rows.append(recovery_row("retrain_ablation", g, r["r2"], r["nrmse"],
                                                r["coverage_90"], full_scores["r2"], params))
                if cfg.R2_SUBSETS == "loo":
                    m_drop = {r["left_out_feature"]: r["r2_drop_mean"] for r in r2_rows
                              if r["model"] == "masked_npe" and r["left_out_feature"]}
                    add_agreement(run, "r2_drop_vs_retrain", m_drop,
                                  {g: float(np.mean(r["r2_drop"])) for g, r in retrain.items()})

    # 5c. single-feature references (marginal importance)
    single = None
    if args.reference == "marginal":
        from references import retrain_single
        print("  reference: single-feature retraining")
        prior_lp = float(problem.prior.log_prob(th_lp).mean())
        single = retrain_single(problem, groups, theta_train, x_train, seed, DEVICE, prior_lp, th_lp, x_lp)
        run["prior_logprob"] = prior_lp
        imp_rows += [{"method": "single_feature_retrain", "feature": g, "estimand": "marginal",
                      "value": r["marginal"]} for g, r in single.items()]
        add_agreement(run, "marginal_vs_single_feature", masked["marginal"],
                      {g: r["marginal"] for g, r in single.items()})

    # 6. log
    curves = [("masked_npe", training_curve(summary))]
    if args.reference in ("full", "ablation"):
        curves.append(("dedicated_npe", ref_info["curve"]))
    if retrain is not None:
        curves += [(f"retrain_without_{g}", r["curve"]) for g, r in retrain.items()]
    if single is not None:
        curves += [(f"single_feature_{g}", r["curve"]) for g, r in single.items()]
    curve_rows = [{"model": name, "epoch": e, "training_loss": tl, "validation_loss": vl}
                  for name, c in curves
                  for e, (tl, vl) in enumerate(zip(c["training_loss"], c["validation_loss"]))]
    folder = rlog.write_seed(output_name(args), cfg, seed, {
        "seeds": [run], "importance": imp_rows, "subset_scores": sub.to_dict("records"),
        "parameter_recovery": r2_rows, "training_curves": curve_rows})
    print(f"  results written to {folder}/")

    # summary
    tab = pd.DataFrame(masked)
    if exact is not None:
        for e in ESTIMANDS:
            tab[f"exact_{e}"] = pd.Series(exact[e])
    if retrain is not None:
        tab["retrain_dlogp"] = pd.Series({g: r["logprob_drop"] for g, r in retrain.items()})
    if single is not None:
        tab["single_marginal"] = pd.Series({g: r["marginal"] for g, r in single.items()})
    if r2_rows and cfg.R2_SUBSETS == "loo":
        tab["masked_dR2"] = pd.Series({r["left_out_feature"]: r["r2_drop_mean"] for r in r2_rows
                                       if r["model"] == "masked_npe" and r["left_out_feature"]})
        if retrain is not None and "r2_drop" in next(iter(retrain.values())):
            tab["retrain_dR2"] = pd.Series({g: float(np.mean(r["r2_drop"])) for g, r in retrain.items()})
    print(tab.loc[groups].round(3).to_string())
    for k in sorted(run):
        if "_vs_" in k or k.endswith(("_all", "_none", "_mean")):
            v = run[k]
            print(f"    {k:36s} {v:.4f}" if isinstance(v, float) else f"    {k:36s} {v}")
    return run


def main():
    sys.stdout.reconfigure(line_buffering=True)   # progress visible in log files as it happens
    args = parse_args()
    cfg = build_config(args)
    toy_kwargs = {"n_channels": args.n_channels, "dup_jitter": args.dup_jitter} if args.simulator == "toy" else {}
    problem = make_problem(args.simulator, cfg, **toy_kwargs)
    print("=" * 74)
    print(f"   masked NPE -- {args.simulator}  seeds={args.seeds}  reference={args.reference}"
          f"  -> results/{output_name(args)}/"
          f"{'  [SMOKE]' if args.smoke else ''}  device={DEVICE}")
    print("=" * 74)
    for seed in args.seeds:
        run_seed(problem, seed, args)


if __name__ == "__main__":
    main()
