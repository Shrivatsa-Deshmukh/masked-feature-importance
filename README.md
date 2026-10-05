# masked-importance

**Which summary statistics actually inform a simulation-based posterior, and by
how much?**

The usual way to answer this is to retrain the posterior estimator once per
feature you remove. This code trains a single **masked neural posterior
estimator (NPE)** instead: during training, random subsets of the summary
statistics are hidden, so afterwards one network can give the posterior for
*any* subset. Feature importance then becomes a quick evaluation instead of
dozens of trainings.

## Why masking

A masked NPE gives the posterior for every subset of summary statistics from a
single training run. All three importance measures (unique, marginal and exact
Shapley) then follow from cheap evaluations rather than one retraining per
subset. Unlike permutation-based shortcuts, subsets
are evaluated within the training distribution. Unlike analytic
marginalization, it works with any flexible posterior estimator.

## How it works

1. **Compute summary statistics** from each simulation and z-score them.
2. **Train with random masks.** Each training example hides a random subset of
   the statistics (set to 0). The network also receives the 0/1 mask itself,
   so it can tell "hidden" from "observed and equal to the average".
3. **Score every subset.** For each subset S of statistics, the score v(S) is
   the average log-probability the network gives the true parameters on
   held-out simulations. Higher means S pins the parameters down better
   (v(S) is a lower bound on the information S carries, in nats).
4. **Read off importance** as differences between scores:

| importance | question it answers | computed as |
|---|---|---|
| **unique**   | What does feature j add when you already have all the others? | v(all) − v(all without j) |
| **marginal** | What does feature j tell you on its own? | v(j alone) − v(nothing) |
| **Shapley**  | What is j's fair share of the total, averaged over every order of adding features? | weighted average over all subsets |

Uncertainty is the spread across independent training runs (seeds).

## Results at a glance

Results are checked against exact values where they are known and against retraining otherwise.

- **Toy problem (exact answers known):** all three importance types are
  recovered closely (mean error 0.02–0.03 nats). The masked NPE is closer to the
  exact values than retraining is, both for unique importance (error 0.030 vs
  0.056) and for marginal importance (0.019 vs 0.053).
- **Jansen-Rit neural-mass model, unique importance:** the ranking matches
  retraining (rank correlation 0.96), and the estimates vary 2–9× less across
  seeds. The values themselves are about 0.7 nats higher than retraining's.
- **Jansen-Rit, marginal importance:** the ranking broadly matches
  single-feature retraining (rank correlation 0.89, with kurtosis misplaced),
  but the values are about 1 nat too low.
- **Cost of masking:** the masked NPE is slightly less accurate than an
  ordinary NPE on the full feature set (R² 0.837 vs 0.851).

**In short:** rankings are reliable; on a hard simulator, absolute values are
biased because the masked network fits rarely seen subsets less well (see
[Lessons learned](#lessons-learned)). Check magnitudes against retraining
before relying on them.

## Detailed results

Values are in nats, mean ± sd over the 9 seeds.

### Toy: linear-Gaussian model with known answers

Eight planted features test the hard cases: a **suppressor** (useless alone,
very informative together), two **near-duplicates** (informative alone,
redundant together), a **null** feature and a pure-**noise** feature (both worth
exactly 0), and weak and large-coefficient features. "Exact" is the true
posterior scored on the same held-out points.

| feature | unique: exact | masked | retrain | marginal: exact | masked | Shapley: exact | masked |
|---|---|---|---|---|---|---|---|
| f1_strong_t1  | 1.939 | 1.846 ± 0.015 | 1.786 | 0.177 | 0.153 | 1.058 | 0.995 |
| f2_strong_t2  | 0.823 | 0.802 ± 0.007 | 0.773 | 0.823 | 0.796 | 0.823 | 0.799 |
| f3_dup_a      | 0.009 | 0.026 ± 0.010 | −0.003 | 0.311 | 0.281 | 0.160 | 0.149 |
| f4_dup_b      | 0.011 | 0.029 ± 0.012 | −0.003 | 0.312 | 0.281 | 0.161 | 0.150 |
| f5_null       | 0 | −0.005 ± 0.005 | −0.028 | 0 | −0.008 | 0 | −0.007 |
| f6_suppressor | 1.763 | 1.672 ± 0.018 | 1.627 | 0 | −0.012 | 0.881 | 0.826 |
| f7_weak       | 0.084 | 0.076 ± 0.009 | 0.052 | 0.144 | 0.133 | 0.114 | 0.104 |
| f8_bigcoef    | 0.276 | 0.272 ± 0.012 | 0.234 | 0.336 | 0.318 | 0.306 | 0.294 |
| noise_0       | 0 | −0.004 ± 0.006 | −0.032 | 0 | −0.007 | 0 | −0.006 |

- Mean error vs exact: unique 0.030 (retraining with one feature removed:
  0.056), marginal 0.019 (retraining on one feature alone: 0.053), Shapley
  0.022. Rank correlation with exact: 0.99, 0.98, 0.99.
- Both retraining references come out slightly low (bias about −0.05 nats,
  e.g. −0.04 for the null and noise features); the masked NPE's bias is about
  −0.02.
- On the full feature set the masked NPE is closer to the true posterior than
  an ordinary NPE (gap 0.22 vs 0.32 nats).

### Jansen-Rit: unique importance vs retraining

A neural-mass model of EEG with 4 parameters, simulated as one channel, with 7
summary statistics. The reference retrains one NPE per removed feature.

| feature | masked unique | retrain unique | masked marginal | masked Shapley |
|---|---|---|---|---|
| total_log_power   | 2.36 ± 0.15 | 1.45 ± 0.53 | 1.61 ± 0.29 | 1.73 ± 0.17 |
| hjorth_mobility   | 2.29 ± 0.08 | 1.55 ± 0.38 | 1.58 ± 0.13 | 1.74 ± 0.08 |
| skewness          | 1.84 ± 0.09 | 0.78 ± 0.52 | 1.03 ± 0.08 | 1.51 ± 0.08 |
| hjorth_complexity | 1.40 ± 0.17 | 0.49 ± 0.35 | 1.29 ± 0.13 | 1.28 ± 0.10 |
| kurtosis          | 0.91 ± 0.10 | 0.34 ± 0.34 | 0.60 ± 0.07 | 0.75 ± 0.06 |
| dominant_freq     | 0.49 ± 0.06 | 0.01 ± 0.45 | 0.90 ± 0.04 | 0.64 ± 0.04 |
| spectral_slope    | 0.44 ± 0.08 | −0.09 ± 0.73 | 0.91 ± 0.04 | 0.55 ± 0.02 |

- Rank correlation with retraining: 0.96 on the seed averages (0.90 ± 0.09 per
  seed). The drop in R² when a feature is left out ranks the features
  similarly (0.75).
- Retraining is too noisy to tell the two weakest features from zero; the
  masked estimates are 2–9× more stable.
- Masked unique values are about 0.7 nats above retraining's.

### Jansen-Rit: marginal importance vs single-feature retraining

The reference trains one NPE on each feature alone.

| feature | masked marginal | single-feature retraining |
|---|---|---|
| total_log_power   | 1.61 ± 0.29 | 3.28 ± 0.27 |
| hjorth_mobility   | 1.58 ± 0.13 | 2.76 ± 0.10 |
| hjorth_complexity | 1.29 ± 0.13 | 2.17 ± 0.04 |
| skewness          | 1.03 ± 0.08 | 2.00 ± 0.05 |
| kurtosis          | 0.60 ± 0.07 | 1.68 ± 0.05 |
| spectral_slope    | 0.91 ± 0.04 | 1.34 ± 0.01 |
| dominant_freq     | 0.90 ± 0.04 | 1.31 ± 0.03 |

- Rank correlation 0.89 (0.83 ± 0.08 per seed); kurtosis is ranked last by the
  masked NPE but above spectral_slope and dominant_freq by retraining.
- Masked values are about 0.95 nats lower (about 57% of the reference), and
  here the single-feature references are the more stable estimate.
- On the toy, single-feature retraining runs slightly *low* (see above), so
  this gap is not the reference overshooting: the masked network fits
  single-feature subsets of Jansen-Rit less well.
- Shapley values mix all subset sizes, so their Jansen-Rit magnitudes carry
  the same caution.

## Getting started

Python 3.11 on Linux, macOS or WSL. Install PyTorch for your platform
(https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
python tests/test_pipeline.py        
```

Run from the repository root:

```bash
# one Jansen-Rit seed, with retraining as the reference
python run.py --simulator jr --seeds 0 --reference ablation

# summarize all seeds in a results folder
python aggregate.py results/jansen_rit
```

Main options (`python run.py --help` for all):

| option | meaning |
|---|---|
| `--simulator {jr, toy}` | Jansen-Rit or the toy |
| `--seeds` | training seeds to run |
| `--reference {none, full, ablation, marginal}` | also train an ordinary NPE (`full`), plus one retrain per removed feature (`ablation`), or one NPE per single feature (`marginal`) |
| `--n_control N` | add N pure-noise control features |
| `--exclude` | features to leave out entirely |
| `--out NAME` | results folder (`results/NAME/`) |
| `--smoke` | tiny, fast run to check that everything works |

**Reproducing everything** (all experiments above, about 9 hours on one GPU):

```bash
bash reproduce_results.sh
python aggregate.py results/jansen_rit results/jansen_rit_marginal results/toy results/toy_marginal
```

## Output files

Each experiment writes to its own folder, `results/<name>/`. Every table is
keyed on the training `seed`; rerunning a seed replaces its rows, and a run
with different settings is refused rather than mixed in.

| file | contents |
|---|---|
| `config.json` | all settings of the experiment |
| `seeds.csv` | one row per seed: accuracy, training, agreement with references |
| `importance.csv` | importance values: seed, method, feature, estimand, value |
| `subset_scores.csv` | the score v(S) of every feature subset |
| `parameter_recovery.csv` | R², error and 90% coverage per parameter, with each feature left out |
| `training_curves.csv` | training and validation loss per epoch |

Names: `masked_npe` is the masked NPE, `dedicated_npe` an ordinary NPE on all
features, `retrain_ablation` an NPE with one feature removed,
`single_feature_retrain` an NPE on one feature alone and `exact` the toy's true
values. In column names, `_all` means every feature, `_none` no features, and
`unique_vs_retrain_spearman` (and similar) is the rank correlation between the
masked NPE and a reference.

## Lessons learned

Mistakes we made along the way, and how the code now avoids them.

**Measuring importance**

- **Shuffling or blanking a feature is misleading.** It creates feature
  combinations the model never saw in training. Masking during training avoids
  this.
- **"Importance" means different things.** Removing a feature measures unique
  importance; using it alone measures marginal importance. They can disagree
  completely: the toy's suppressor is useless alone and essential together. Always say which one you report.
- **Where the masked NPE's errors come from.** An importance value is a
  difference of two subset scores, so its error is the difference in how well
  the network fits those two subsets. With uniform masks, the full feature set
  is seen about 7× more often in training than any one leave-one-out subset, so
  unique values come out too high; single-feature subsets are fitted worse
  than the empty set (which is just the prior), so marginal values come out
  too low. Showing the full set more often makes this worse, not better. The
  toy didn't reveal either effect because its posteriors are simple Gaussians.

**Working with sbi**

- **Double input scaling.** sbi standardizes inputs before the embedding
  network by default. Our feature network already z-scores with its own
  statistics, so the two clashed and the network saw inputs far from the
  expected range. Turn sbi's scaling off (`z_score_x="none"`).
- **Masks set on a copy.** sbi copies the embedding network internally, so a
  mask set on your own copy is silently ignored. Masks are now set through the
  posterior, which updates every copy and fails loudly if it finds none.
- **Probability outside the prior.** The flow can put some probability where
  the parameters can never be (about 6% for Jansen-Rit's box prior). Scores are
  renormalized to the prior box.
- **Training cut off at the epoch limit.** If the maximum number of epochs is
  reached, sbi keeps the last weights, not the best ones. Make sure models
  stop on their own before the limit.

**Simulator and features**

- **The DC offset leaked into the spectrum**, pinning about a third of the
  dominant-frequency values to the lowest bin. The spectrum is now computed on
  the mean-removed signal.
- **Features real recordings can't provide.** The signal's mean (DC level) is
  very informative in the simulator but not measurable in real EEG, which is
  high-pass filtered. It is excluded by default.
- **Noise scaled to the wrong reference.** Observation noise was set relative
  to the signal level *including* its DC offset, so traces with a large offset
  got far too much noise. Observation noise is now off by default (see the
  note in `config.py`).
- **Unseeded randomness.** Part of the simulated noise ignored the seed, so
  "the same seed" gave different data. All random sources are now seeded and
  tested.

**Training and evaluation**

- **Validation masks redrawn every epoch** made early stopping noisier; each
  validation point now keeps a fixed mask.
- **Comparing with the wrong truth.** Comparing averages over a fixed test set
  with theoretical expected values mixed in sampling noise and produced
  impossible negative gaps. The toy's exact posterior is now scored on the same
  points.

## Repository layout

```
run.py                 runs one experiment (training, scoring, references)
config.py              all settings
masked_npe.py          the masked network, its masks and training
importance.py          scores every subset; unique, marginal and Shapley values
evaluation.py          log-probabilities, R², coverage, agreement statistics
references.py          the retraining references
features.py            summary statistics and z-scoring
problems.py            the two problems behind one interface
jansen_rit.py          Jansen-Rit simulator
forward_model.py       optional EEG forward model (needs MNE-Python)
toy.py                 the toy model and its exact answers
results_log.py         writes the results folders
aggregate.py           summarizes the seeds of a results folder
reproduce_results.sh   reruns every experiment
tests/                 checks of the simulators, features, masking and maths
```
