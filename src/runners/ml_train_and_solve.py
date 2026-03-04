#!/usr/bin/env python3
"""
ML-Assisted UC: training, evaluation, and comparison with standard MILP.

Usage examples
--------------
# Train on rts_gmlc with 5 augmentations per instance, then evaluate
python src/runners/ml_train_and_solve.py --data-dir examples/rts_gmlc --n-augmented 5

# Use pre-trained models
python src/runners/ml_train_and_solve.py --data-dir examples/rts_gmlc --load-models ml_models/rts_gmlc.pkl

# Train only (no evaluation)
python src/runners/ml_train_and_solve.py --data-dir examples/rts_gmlc --train-only --save-models ml_models/rts_gmlc.pkl

# Leave-one-out cross-validation
python src/runners/ml_train_and_solve.py --data-dir examples/rts_gmlc --cross-validate

# Full pipeline with confidence threshold
python src/runners/ml_train_and_solve.py --data-dir examples/rts_gmlc --confidence 0.9
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# запустить на больших инстансах

# подумать как ускорить генерацию данных

# Поддерживаемые ML методы: lr (логистическая регрессия), rf (random forest), cb (CatBoost)

# попробовать параллельный запуск - стандартный и ml одновременно, ограничиваем время и в любой момент времени понимаем
# какое решение, до первой остановки, если нет, то видим текущее решение и оценку оптимума. и если разница минимальная,
# то оба останавливаем

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.solvers.ml_uc_solver import (
    extract_features,
    generate_training_data,
    train_commitment_models,
    save_models,
    load_models,
    solve_ml_assisted,
    solve_standard,
    solve_instance_for_labels,
    augment_instance,
    _cache_path_for,
    _remap_labels,
)


def _resolve_cache_dir(args):
    """Return the cache directory path, or None if caching is disabled."""
    if args.no_cache:
        return None
    if args.cache_dir:
        return args.cache_dir
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    return os.path.join(project_root, "training_cache", dataset_name)


def train_and_evaluate(args):
    """Train ML models and compare ML-assisted vs standard solving."""
    data_dir = os.path.join(project_root, args.data_dir)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))

    if not files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Dataset: {args.data_dir} ({len(files)} instances)")
    print(f"Solver: {args.solver}, gap: {args.gap}, threads: {args.threads}, "
          f"ML method: {args.ml_method}")
    if args.first_feasible:
        print(f"Training mode: FIRST FEASIBLE (fast labels)")

    cache_dir = _resolve_cache_dir(args)
    if cache_dir:
        print(f"Cache dir: {cache_dir}")
    print()

    # -------------------------------------------------------------------
    # Phase 1: Get or train ML models
    # -------------------------------------------------------------------
    if args.load_models:
        print(f"Loading pre-trained models from {args.load_models}")
        models_dict, T = load_models(args.load_models)
        print(f"  {len(models_dict)} generators with models, "
              f"{T} time periods loaded")
    else:
        print("=" * 70)
        print("PHASE 1: Generating training data")
        print("=" * 70)

        t0 = time.time()
        X, Y, gen_names, T = generate_training_data(
            data_dir,
            solver_name=args.solver,
            gap=args.train_gap,
            time_limit=args.time_limit,
            threads=args.threads,
            n_augmented=args.n_augmented,
            noise_std=args.noise_std,
            scale_range=(args.scale_min, args.scale_max),
            cache_dir=cache_dir,
            first_feasible=args.first_feasible,
            verbose=True,
        )
        train_data_time = time.time() - t0
        print(f"\nTraining data generated in {train_data_time:.1f}s")

        print("\n" + "=" * 70)
        print("PHASE 2: Training ML models")
        print("=" * 70)

        t0 = time.time()
        models_dict, scores = train_commitment_models(
            X, Y, gen_names, T, method=args.ml_method, verbose=True)
        train_time = time.time() - t0
        print(f"Training completed in {train_time:.1f}s")

        # Save models if requested
        if args.save_models:
            save_models(models_dict, T, args.save_models)
            print(f"Models saved to {args.save_models}")

    if args.train_only:
        print("\n--train-only specified, skipping evaluation.")
        return

    # -------------------------------------------------------------------
    # Phase 2 / 3: Evaluate ML-assisted vs standard solving
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: Evaluating ML-assisted vs standard solving")
    print("=" * 70)

    results = []
    for fi, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        data = json.load(open(fpath))

        print(f"\n--- [{fi + 1}/{len(files)}] {fname} ---")

        # Standard solve
        print("  Standard MILP:")
        std_result = solve_standard(
            data, solver_name=args.solver, gap=args.gap,
            time_limit=args.time_limit, threads=args.threads,
            verbose=True,
        )

        # ML-assisted solve
        print("  ML-assisted MILP:")
        ml_result = solve_ml_assisted(
            data, models_dict, T,
            solver_name=args.solver, gap=args.gap,
            time_limit=args.time_limit, threads=args.threads,
            confidence_threshold=args.confidence,
            verbose=True,
        )

        # Compare
        if std_result["objective"] and ml_result["objective"]:
            obj_gap = abs(ml_result["objective"] - std_result["objective"]) / abs(
                std_result["objective"]) * 100
            speedup = std_result["solve_time"] / max(ml_result["solve_time"], 0.01)
        else:
            obj_gap = None
            speedup = None

        results.append({
            "instance": fname,
            "std_obj": std_result["objective"],
            "ml_obj": ml_result["objective"],
            "std_time": std_result["solve_time"],
            "ml_time": ml_result["solve_time"],
            "speedup": speedup,
            "obj_gap_pct": obj_gap,
            "std_status": std_result["status"],
            "ml_status": ml_result["status"],
            "n_fixed_off": ml_result["reduction_stats"]["n_fixed_off"],
            "n_fixed_on": ml_result["reduction_stats"]["n_fixed_on"],
            "n_free": ml_result["reduction_stats"]["n_free"],
            "vars_fixed_pct": 100 * ml_result["reduction_stats"]["n_vars_fixed"] /
                              max(ml_result["reduction_stats"]["n_vars_total"], 1),
        })

        if speedup is not None:
            print(f"  => Speedup: {speedup:.2f}x, "
                  f"Obj gap: {obj_gap:.4f}%, "
                  f"Vars fixed: {results[-1]['vars_fixed_pct']:.1f}%")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Instance':<30} {'Std time':>10} {'ML time':>10} "
          f"{'Speedup':>8} {'Obj gap%':>10} {'Fixed%':>8}")
    print("-" * 76)

    speedups = []
    gaps = []
    for r in results:
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
        og = f"{r['obj_gap_pct']:.4f}" if r["obj_gap_pct"] is not None else "N/A"
        vf = f"{r['vars_fixed_pct']:.1f}"
        print(f"{r['instance']:<30} {r['std_time']:>10.2f} {r['ml_time']:>10.2f} "
              f"{sp:>8} {og:>10} {vf:>8}")
        if r["speedup"] is not None:
            speedups.append(r["speedup"])
        if r["obj_gap_pct"] is not None:
            gaps.append(r["obj_gap_pct"])

    if speedups:
        print("-" * 76)
        print(f"{'AVERAGE':<30} "
              f"{'':>10} {'':>10} "
              f"{np.mean(speedups):>7.2f}x "
              f"{np.mean(gaps):>10.4f} ")
        print(f"\nMean speedup: {np.mean(speedups):.2f}x")
        print(f"Mean obj gap: {np.mean(gaps):.4f}%")
        print(f"Max  obj gap: {np.max(gaps):.4f}%")

    # Save results to CSV
    if args.output:
        import csv
        fieldnames = list(results[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")


def cross_validate(args):
    """Leave-one-out cross-validation: train on N-1 instances, test on 1."""
    data_dir = os.path.join(project_root, args.data_dir)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))

    if len(files) < 2:
        print("Need at least 2 instances for cross-validation")
        return

    print(f"Leave-one-out cross-validation on {len(files)} instances")
    print(f"Augmentation: {args.n_augmented} per instance, "
          f"noise_std={args.noise_std}")

    cache_dir = _resolve_cache_dir(args)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cache dir: {cache_dir}")
    print()

    all_datasets = {}
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        all_datasets[fname] = json.load(open(fpath))

    # Collect union of all generator names and T
    all_gen_names = set()
    T = None
    for data in all_datasets.values():
        all_gen_names.update(data["thermal_generators"].keys())
        if T is None:
            T = data["time_periods"]
    gen_names = sorted(all_gen_names)

    print(f"Union of generators: {len(gen_names)} across {len(files)} instances")

    # Pre-solve all instances (original + augmented) with caching.
    # Each instance is solved once and reused across folds.
    print("=" * 60)
    print("Pre-solving all instances")
    print("=" * 60)
    solutions = {}  # (fname, tag) -> (features, labels_in_union_order)
    n_cached = 0

    for fi, fname in enumerate(files):
        data = all_datasets[fname]
        inst_gen_names = sorted(data["thermal_generators"].keys())

        instances_to_solve = [("original", data)]
        rng = np.random.RandomState(hash(fname) % (2 ** 31))
        for ai in range(args.n_augmented):
            scale = rng.uniform(args.scale_min, args.scale_max)
            aug = augment_instance(data, scale_factor=scale,
                                   noise_std=args.noise_std,
                                   seed=rng.randint(2 ** 31))
            instances_to_solve.append((f"aug-{ai}", aug))

        for tag, inst_data in instances_to_solve:
            label = f"[{fi + 1}/{len(files)}] {fname} ({tag})"

            # Try loading from cache
            if cache_dir:
                cp = _cache_path_for(cache_dir, fname, tag)
                if os.path.exists(cp):
                    cached = np.load(cp, allow_pickle=True)
                    if "gen_names" in cached:
                        cached_gn = list(cached["gen_names"])
                        labels = _remap_labels(
                            cached_gn, cached["labels"], gen_names, T)
                        solutions[(fname, tag)] = (cached["features"], labels)
                        n_cached += 1
                        print(f"  {label} ... CACHED")
                        continue
                    else:
                        # Old cache format — delete and re-solve
                        print(f"  {label} ... old cache format, re-solving")
                        os.remove(cp)

            print(f"  Solving {label} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                commitment, obj = solve_instance_for_labels(
                    inst_data, args.solver, args.train_gap,
                    args.time_limit, args.threads,
                    first_feasible=args.first_feasible)
                features = extract_features(inst_data)

                # Build labels in union gen order; NaN for missing generators
                labels = np.full(len(gen_names) * T, np.nan)
                for gi, g in enumerate(gen_names):
                    if g in commitment:
                        for t_idx in range(T):
                            labels[gi * T + t_idx] = commitment[g][t_idx]

                solutions[(fname, tag)] = (features, labels)

                # Save to cache with instance gen_names
                if cache_dir:
                    inst_labels = []
                    for g in inst_gen_names:
                        inst_labels.extend(commitment[g])
                    np.savez(cp, features=features,
                             labels=np.array(inst_labels),
                             gen_names=np.array(inst_gen_names))

                elapsed = time.time() - t0
                print(f"OK  obj={obj:.0f}  time={elapsed:.1f}s")
            except Exception as e:
                print(f"FAILED: {e}")

    print(f"\nPre-solved: {len(solutions)} samples"
          f" ({n_cached} from cache)")

    # Run cross-validation folds
    results = []
    all_tags = ["original"] + [f"aug-{ai}" for ai in range(args.n_augmented)]

    for test_idx, test_fname in enumerate(files):
        print(f"\n{'=' * 60}")
        print(f"Fold {test_idx + 1}/{len(files)}: testing on {test_fname}")
        print(f"{'=' * 60}")

        # Assemble training data from pre-solved instances
        train_files = [f for f in files if f != test_fname]

        X_list, Y_list = [], []
        for fname in train_files:
            for tag in all_tags:
                if (fname, tag) in solutions:
                    features, labels = solutions[(fname, tag)]
                    X_list.append(features)
                    Y_list.append(labels)

        if len(X_list) < 2:
            print("  Not enough training samples, skipping fold")
            continue

        X = np.array(X_list)
        Y = np.array(Y_list)
        print(f"  Training data: {X.shape[0]} samples")

        models_dict, scores = train_commitment_models(
            X, Y, gen_names, T, method=args.ml_method, verbose=True)

        # Test on held-out instance
        test_data = all_datasets[test_fname]

        print(f"\n  Evaluating on {test_fname}:")

        # Standard solve
        std_result = solve_standard(
            test_data, solver_name=args.solver, gap=args.gap,
            time_limit=args.time_limit, threads=args.threads, verbose=True)

        # ML-assisted solve
        ml_result = solve_ml_assisted(
            test_data, models_dict, T,
            solver_name=args.solver, gap=args.gap,
            time_limit=args.time_limit, threads=args.threads,
            confidence_threshold=args.confidence, verbose=True)

        if std_result["objective"] and ml_result["objective"]:
            obj_gap = abs(ml_result["objective"] - std_result["objective"]) / abs(
                std_result["objective"]) * 100
            speedup = std_result["solve_time"] / max(ml_result["solve_time"], 0.01)
        else:
            obj_gap = None
            speedup = None

        results.append({
            "fold": test_idx + 1,
            "test_instance": test_fname,
            "n_train_samples": X.shape[0],
            "std_obj": std_result["objective"],
            "ml_obj": ml_result["objective"],
            "std_time": std_result["solve_time"],
            "ml_time": ml_result["solve_time"],
            "speedup": speedup,
            "obj_gap_pct": obj_gap,
            "vars_fixed_pct": 100 * ml_result["reduction_stats"]["n_vars_fixed"] /
                              max(ml_result["reduction_stats"]["n_vars_total"], 1),
        })

        if speedup is not None:
            print(f"\n  => Speedup: {speedup:.2f}x, Obj gap: {obj_gap:.4f}%")

    # Summary
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    speedups = [r["speedup"] for r in results if r["speedup"] is not None]
    gaps = [r["obj_gap_pct"] for r in results if r["obj_gap_pct"] is not None]

    for r in results:
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
        og = f"{r['obj_gap_pct']:.4f}%" if r["obj_gap_pct"] is not None else "N/A"
        print(f"  Fold {r['fold']}: {r['test_instance']:<25} "
              f"speedup={sp:<8} obj_gap={og:<12} fixed={r['vars_fixed_pct']:.1f}%")

    if speedups:
        print(f"\n  Mean speedup: {np.mean(speedups):.2f}x")
        print(f"  Mean obj gap: {np.mean(gaps):.4f}%")
        print(f"  Max  obj gap: {np.max(gaps):.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description="ML-Assisted UC: train, evaluate, and compare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to dataset directory (e.g. examples/rts_gmlc)")
    parser.add_argument("--ml-method", default="lr",
                        choices=["lr", "rf", "cb"],
                        help="ML method: lr (logistic regression), "
                             "rf (random forest), cb (CatBoost) "
                             "(default: lr)")
    parser.add_argument("--solver", default="appsi_highs",
                        help="Solver name (default: appsi_highs)")
    parser.add_argument("--gap", type=float, default=0.01,
                        help="MIP gap for evaluation solves (default: 0.01)")
    parser.add_argument("--train-gap", type=float, default=0.02,
                        help="MIP gap for training data generation (default: 0.02)")
    parser.add_argument("--time-limit", type=int, default=3000,
                        help="Solver time limit in seconds (default: 300)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Solver threads (default: 1)")
    parser.add_argument("--n-augmented", type=int, default=0,
                        help="Augmented instances per original (default: 5)")
    parser.add_argument("--noise-std", type=float, default=0.03,
                        help="Noise std-dev for augmentation (default: 0.03)")
    parser.add_argument("--scale-min", type=float, default=0.9,
                        help="Min demand scale factor (default: 0.9)")
    parser.add_argument("--scale-max", type=float, default=1.1,
                        help="Max demand scale factor (default: 1.1)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Confidence threshold for fixing (e.g. 0.9)")
    parser.add_argument("--save-models", default=None,
                        help="Save trained models to this path")
    parser.add_argument("--load-models", default=None,
                        help="Load pre-trained models from this path")
    parser.add_argument("--train-only", action="store_true",
                        help="Only generate training data and train models")
    parser.add_argument("--cross-validate", action="store_true",
                        help="Run leave-one-out cross-validation")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to cache solved instances "
                             "(default: training_cache/<dataset>/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable training data caching")
    parser.add_argument("--first-feasible", action="store_true",
                        help="Stop at first feasible solution when generating "
                             "training data (much faster for hard instances)")
    parser.add_argument("--output", default=None,
                        help="Save comparison results to CSV")

    args = parser.parse_args()

    if args.cross_validate:
        cross_validate(args)
    else:
        train_and_evaluate(args)


if __name__ == "__main__":
    main()
