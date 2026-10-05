#!/usr/bin/env bash
# Uniform masks, 8192 simulations, 9 seeds:
#   1. Jansen-Rit with a seed-matched full-feature NPE and retrain-ablation
#      references                                   -> results/jansen_rit/
#   2. Jansen-Rit with single-feature references, the check on marginal
#      importance (R^2 skipped, log-prob only)       -> results/jansen_rit_marginal/
#   3. the toy with retrain-ablation references      -> results/toy/
#   4. the toy with single-feature references        -> results/toy_marginal/
# Rerunning replaces each seed's rows in place.
# Set PYTHON to choose the interpreter (default: python).
set -u
cd "$(dirname "$0")"          # paths below are relative to the repository root
PY="${PYTHON:-python}"
SEEDS="0 1 5 10 15 50 55 99 105"
N=8192
mkdir -p results/logs
for S in $SEEDS; do
  "$PY" run.py --simulator jr --seeds $S --num_simulations $N --reference ablation \
     --out jansen_rit > results/logs/jr_s$S.log 2>&1 \
     && echo "JR SEED $S DONE" || echo "JR SEED $S FAILED"
done
for S in $SEEDS; do
  "$PY" run.py --simulator jr --seeds $S --num_simulations $N --reference marginal --r2 none \
     --out jansen_rit_marginal > results/logs/marginal_jr_s$S.log 2>&1 \
     && echo "MARGINAL JR SEED $S DONE" || echo "MARGINAL JR SEED $S FAILED"
done
for S in $SEEDS; do
  "$PY" run.py --simulator toy --seeds $S --num_simulations $N --n_control 1 --reference ablation \
     --out toy > results/logs/toy_s$S.log 2>&1 \
     && echo "TOY SEED $S DONE" || echo "TOY SEED $S FAILED"
done
for S in $SEEDS; do
  "$PY" run.py --simulator toy --seeds $S --num_simulations $N --n_control 1 --reference marginal --r2 none \
     --out toy_marginal > results/logs/marginal_toy_s$S.log 2>&1 \
     && echo "MARGINAL TOY SEED $S DONE" || echo "MARGINAL TOY SEED $S FAILED"
done
echo "ALL DONE"
