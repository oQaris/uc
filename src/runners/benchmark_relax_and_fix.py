"""
Benchmark script for comparing Relax-and-Fix algorithm with standard HiGHS solver

This script:
1. Runs solve_relax_and_fix on each example from examples/
2. Measures solve time
3. Runs standard HiGHS solver with same time limit
4. Compares results (time, objective value)
5. Handles exceptions and skips failed instances
6. Outputs results to CSV file
7. Supports multiprocessing (one thread per instance)
8. Finds optimal hyperparameters for solve_relax_and_fix
"""

import json
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import csv
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.uc_model import build_uc_model
from solvers.relax_and_fix import solve_relax_and_fix
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.environ import value


def solve_with_standard_highs(data: dict, time_limit: float, gap: float = 0.01) -> Dict:
    """
    Solve UC problem with standard HiGHS solver

    Args:
        data: Problem data dictionary
        time_limit: Time limit in seconds
        gap: MIP gap tolerance

    Returns:
        dict with solve_time, objective, status, termination_condition
    """
    try:
        model = build_uc_model(data)
        solver = SolverFactory("appsi_highs")

        # Set solver options via highs_options (more reliable than config)
        # IMPORTANT: solver.config.time_limit doesn't work for appsi_highs!
        solver.highs_options['mip_rel_gap'] = gap
        solver.highs_options['time_limit'] = time_limit

        start_time = time.time()
        result = solver.solve(model)
        actual_solve_time = time.time() - start_time

        termination = result.solver.termination_condition

        # Extract objective if solution exists
        objective = None
        if termination in [TerminationCondition.optimal, TerminationCondition.feasible]:
            try:
                objective = value(model.obj)
            except:
                objective = None

        return {
            'solve_time': actual_solve_time,
            'objective': objective,
            'status': str(termination),
            'termination_condition': termination
        }
    except Exception as e:
        return {
            'solve_time': None,
            'objective': None,
            'status': 'error',
            'error': str(e),
            'termination_condition': None
        }


def solve_with_relax_and_fix(data: dict,
                             window_size: int = 8,
                             window_step: int = 4,
                             gap: float = 0.01,
                             generators_per_iteration: Optional[int] = None,
                             use_limited_horizon: bool = True,
                             verbose: bool = False) -> Dict:
    """
    Solve UC problem with Relax-and-Fix algorithm

    Args:
        data: Problem data dictionary
        window_size: Size of time window for binary variables
        window_step: Step size for moving window
        gap: MIP gap tolerance
        generators_per_iteration: Number of generators per iteration (None = all)
        use_limited_horizon: Use adaptive lookahead window
        verbose: Show solver output

    Returns:
        dict with solve_time, objective, status, verified_objective, feasible
    """
    try:
        model = build_uc_model(data)

        result = solve_relax_and_fix(
            model=model,
            window_size=window_size,
            window_step=window_step,
            gap=gap,
            solver_name="appsi_highs",
            verbose=verbose,
            verify_solution=True,
            data=data,
            model_builder=build_uc_model,
            generators_per_iteration=generators_per_iteration,
            use_limited_horizon=use_limited_horizon
        )

        return result
    except Exception as e:
        return {
            'solve_time': None,
            'objective': None,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def benchmark_single_instance(instance_path: str,
                              window_size: int = 8,
                              window_step: int = 4,
                              gap: float = 0.01,
                              generators_per_iteration: Optional[int] = None,
                              use_limited_horizon: bool = True,
                              verbose: bool = False) -> Dict:
    """
    Run benchmark on a single instance

    Args:
        instance_path: Path to JSON instance file
        window_size: Relax-and-Fix window size
        window_step: Relax-and-Fix window step
        gap: MIP gap tolerance
        generators_per_iteration: Generators per R&F iteration
        use_limited_horizon: Use adaptive lookahead
        verbose: Show detailed output

    Returns:
        dict with benchmark results or None if instance failed
    """
    instance_name = Path(instance_path).stem
    dataset = Path(instance_path).parent.name

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {dataset}/{instance_name}")
        print(f"{'=' * 60}")

    try:
        # Load data
        with open(instance_path, 'r') as f:
            data = json.load(f)

        num_thermal_gens = len(data.get('thermal_generators', {}))
        num_periods = data.get('time_periods', 0)

        if verbose:
            print(f"Instance info: {num_thermal_gens} generators, {num_periods} periods")

        # Run Relax-and-Fix
        if verbose:
            print("\nRunning Relax-and-Fix...")

        rf_result = solve_with_relax_and_fix(
            data=data,
            window_size=window_size,
            window_step=window_step,
            gap=gap,
            generators_per_iteration=generators_per_iteration,
            use_limited_horizon=use_limited_horizon,
            verbose=verbose
        )

        # Check if R&F failed
        if rf_result.get('status') == 'error':
            if verbose:
                print(f"Relax-and-Fix failed: {rf_result.get('error')}")
            return None

        rf_time = rf_result['solve_time']
        rf_objective = rf_result.get('verified_objective') or rf_result.get('objective')
        rf_feasible = rf_result.get('feasible', True)

        if verbose:
            print(f"R&F completed in {rf_time:.2f}s, objective={rf_objective:.2f}, feasible={rf_feasible}")

        # Skip if R&F solution is infeasible
        if not rf_feasible:
            if verbose:
                print("R&F solution is infeasible, skipping HiGHS comparison")
            return None

        # Run standard HiGHS with same time limit
        if verbose:
            print(f"\nRunning HiGHS with {rf_time:.2f}s time limit...")

        highs_result = solve_with_standard_highs(
            data=data,
            time_limit=rf_time,
            gap=gap
        )

        # Check if HiGHS failed
        if highs_result.get('status') == 'error':
            if verbose:
                print(f"HiGHS failed: {highs_result.get('error')}")
            return None

        highs_time = highs_result['solve_time']
        highs_objective = highs_result['objective']
        highs_status = highs_result['status']

        if verbose:
            print(f"HiGHS completed in {highs_time:.2f}s, objective={highs_objective}, status={highs_status}")

        # Calculate gap if both solutions exist
        gap_percent = None
        if rf_objective is not None and highs_objective is not None and highs_objective > 0:
            gap_percent = ((rf_objective - highs_objective) / (highs_objective + rf_objective)) * 200

        result = {
            'dataset': dataset,
            'instance': instance_name,
            'num_generators': num_thermal_gens,
            'num_periods': num_periods,
            'rf_time': rf_time,
            'rf_objective': rf_objective,
            'rf_feasible': rf_feasible,
            'highs_time': highs_time,
            'highs_objective': highs_objective,
            'highs_status': highs_status,
            'gap_percent': gap_percent,
            'window_size': window_size,
            'window_step': window_step,
            'generators_per_iteration': generators_per_iteration,
            'use_limited_horizon': use_limited_horizon
        }

        if verbose:
            print(f"\nResults:")
            print(f"  R&F:   {rf_time:.2f}s, obj={rf_objective:.2f}")
            print(f"  HiGHS: {highs_time:.2f}s, obj={highs_objective if highs_objective else 'N/A'}")
            if gap_percent is not None:
                print(f"  Gap:   {gap_percent:+.2f}%")

        return result

    except Exception as e:
        if verbose:
            print(f"Error processing instance: {e}")
            traceback.print_exc()
        return None


def run_benchmark(instances: List[str],
                  output_file: str = "benchmark_results.csv",
                  max_workers: int = 4,
                  window_size: int = 8,
                  window_step: int = 4,
                  gap: float = 0.01,
                  generators_per_iteration: Optional[int] = None,
                  use_limited_horizon: bool = True,
                  verbose: bool = False) -> List[Dict]:
    """
    Run benchmark on multiple instances in parallel

    Args:
        instances: List of instance file paths
        output_file: Output CSV file path
        max_workers: Number of parallel workers
        window_size: Relax-and-Fix window size
        window_step: Relax-and-Fix window step
        gap: MIP gap tolerance
        generators_per_iteration: Generators per R&F iteration
        use_limited_horizon: Use adaptive lookahead
        verbose: Show detailed output

    Returns:
        List of benchmark results
    """
    results = []

    print(f"Running benchmark on {len(instances)} instances with {max_workers} workers...")
    print(f"Hyperparameters: window_size={window_size}, window_step={window_step}, "
          f"generators_per_iteration={generators_per_iteration}, use_limited_horizon={use_limited_horizon}")

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(
                benchmark_single_instance,
                instance,
                window_size,
                window_step,
                gap,
                generators_per_iteration,
                use_limited_horizon,
                verbose
            ): instance
            for instance in instances
        }

        # Collect results as they complete
        for future in as_completed(future_to_instance):
            instance = future_to_instance[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    print(f"✓ Completed {result['dataset']}/{result['instance']}: "
                          f"R&F={result['rf_time']:.1f}s, HiGHS obj={result['highs_objective']}")
                else:
                    print(f"✗ Skipped {Path(instance).parent.name}/{Path(instance).stem} (error or infeasible)")
            except Exception as e:
                print(f"✗ Failed {Path(instance).parent.name}/{Path(instance).stem}: {e}")

    # Write results to CSV
    if results:
        write_results_to_csv(results, output_file)
        print(f"\nResults written to {output_file}")
        print_summary_statistics(results)
    else:
        print("\nNo successful results to write")

    return results


def write_results_to_csv(results: List[Dict], output_file: str):
    """Write benchmark results to CSV file"""
    if not results:
        return

    fieldnames = [
        'dataset', 'instance', 'num_generators', 'num_periods',
        'rf_time', 'rf_objective', 'rf_feasible',
        'highs_time', 'highs_objective', 'highs_status',
        'gap_percent',
        'window_size', 'window_step', 'generators_per_iteration', 'use_limited_horizon'
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary_statistics(results: List[Dict]):
    """Print summary statistics of benchmark results"""
    if not results:
        return

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Overall statistics
    total_instances = len(results)
    avg_rf_time = sum(r['rf_time'] for r in results) / total_instances
    avg_highs_time = sum(r['highs_time'] for r in results) / total_instances

    # Count HiGHS solution quality
    highs_optimal = sum(1 for r in results if 'optimal' in r['highs_status'].lower())
    highs_feasible = sum(1 for r in results if 'feasible' in r['highs_status'].lower())

    # Calculate average gap for instances where both have objectives
    gaps = [r['gap_percent'] for r in results if r['gap_percent'] is not None]
    avg_gap = sum(gaps) / len(gaps) if gaps else None

    print(f"Total instances: {total_instances}")
    print(f"Average R&F time: {avg_rf_time:.2f}s")
    print(f"Average HiGHS time: {avg_highs_time:.2f}s")
    print(f"HiGHS optimal solutions: {highs_optimal}/{total_instances}")
    print(f"HiGHS feasible solutions: {highs_feasible}/{total_instances}")
    if avg_gap is not None:
        print(f"Average objective gap: {avg_gap:+.2f}%")

    # Per-dataset statistics
    print("\nPer-dataset statistics:")
    datasets = set(r['dataset'] for r in results)
    for dataset in sorted(datasets):
        dataset_results = [r for r in results if r['dataset'] == dataset]
        if not dataset_results:
            continue

        count = len(dataset_results)
        avg_time = sum(r['rf_time'] for r in dataset_results) / count
        dataset_gaps = [r['gap_percent'] for r in dataset_results if r['gap_percent'] is not None]
        avg_dataset_gap = sum(dataset_gaps) / len(dataset_gaps) if dataset_gaps else None

        print(f"  {dataset}: {count} instances, avg R&F time={avg_time:.2f}s", end='')
        if avg_dataset_gap is not None:
            print(f", avg gap={avg_dataset_gap:+.2f}%")
        else:
            print()


def optimize_hyperparameters(instances: List[str],
                             output_dir: str = "hyperparameter_optimization",
                             max_workers: int = 4,
                             sample_size: int = 10) -> Dict:
    """
    Find optimal hyperparameters for solve_relax_and_fix using grid search

    Args:
        instances: List of instance file paths
        output_dir: Directory to save optimization results
        max_workers: Number of parallel workers
        sample_size: Number of instances to sample for optimization

    Returns:
        dict with best hyperparameters and results
    """
    import random
    from pathlib import Path

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Sample instances for faster optimization
    if len(instances) > sample_size:
        sampled_instances = random.sample(instances, sample_size)
        print(f"Sampling {sample_size} instances out of {len(instances)} for hyperparameter optimization")
    else:
        sampled_instances = instances
        print(f"Using all {len(instances)} instances for hyperparameter optimization")

    # Define hyperparameter grid
    hyperparameter_grid = {
        'window_size': [6, 8, 12],
        'window_step': [3, 4, 6],
        'use_limited_horizon': [True, False],
        'generators_per_iteration': [None, 50, 100]  # None means all generators
    }

    # Generate all combinations
    from itertools import product

    param_combinations = []
    for ws, wstep, ulh, gpi in product(
            hyperparameter_grid['window_size'],
            hyperparameter_grid['window_step'],
            hyperparameter_grid['use_limited_horizon'],
            hyperparameter_grid['generators_per_iteration']
    ):
        # Skip invalid combinations (step > size)
        if wstep > ws:
            continue
        param_combinations.append({
            'window_size': ws,
            'window_step': wstep,
            'use_limited_horizon': ulh,
            'generators_per_iteration': gpi
        })

    print(f"\nTesting {len(param_combinations)} hyperparameter combinations...")

    best_params = None
    best_score = float('inf')  # Lower is better (avg solve time)
    all_optimization_results = []

    for i, params in enumerate(param_combinations, 1):
        print(f"\n[{i}/{len(param_combinations)}] Testing: {params}")

        output_file = f"{output_dir}/config_{i}.csv"

        results = run_benchmark(
            instances=sampled_instances,
            output_file=output_file,
            max_workers=max_workers,
            **params,
            gap=0.01,
            verbose=False
        )

        if not results:
            print("  No successful results, skipping")
            continue

        # Calculate score (average solve time + penalty for infeasible solutions)
        avg_time = sum(r['rf_time'] for r in results) / len(results)
        infeasible_count = sum(1 for r in results if not r['rf_feasible'])
        infeasible_penalty = infeasible_count * 1000  # Heavy penalty for infeasible solutions

        score = avg_time + infeasible_penalty

        optimization_result = {
            **params,
            'avg_solve_time': avg_time,
            'successful_instances': len(results),
            'infeasible_count': infeasible_count,
            'score': score
        }
        all_optimization_results.append(optimization_result)

        print(f"  Score: {score:.2f} (avg_time={avg_time:.2f}s, infeasible={infeasible_count})")

        if score < best_score:
            best_score = score
            best_params = params
            print(f"  *** NEW BEST ***")

    # Save optimization summary
    summary_file = f"{output_dir}/optimization_summary.csv"
    if all_optimization_results:
        fieldnames = ['window_size', 'window_step', 'use_limited_horizon', 'generators_per_iteration',
                      'avg_solve_time', 'successful_instances', 'infeasible_count', 'score']
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(all_optimization_results, key=lambda x: x['score']))

        print(f"\nOptimization summary saved to {summary_file}")

    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS")
    print("=" * 60)
    if best_params:
        for key, value in best_params.items():
            print(f"{key}: {value}")
        print(f"Score: {best_score:.2f}")
    else:
        print("No successful configuration found")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_optimization_results
    }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark Relax-and-Fix vs HiGHS solver')
    parser.add_argument('--examples-dir', default=r'C:\Users\oQaris\Desktop\Git\uc\examples',
                        help='Directory containing example instances')
    parser.add_argument('--output', default='benchmark_results.csv',
                        help='Output CSV file')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of parallel workers')
    parser.add_argument('--window-size', type=int, default=8,
                        help='Relax-and-Fix window size')
    parser.add_argument('--window-step', type=int, default=8,
                        help='Relax-and-Fix window step')
    parser.add_argument('--generators-per-iteration', type=int, default=None,
                        help='Generators per iteration (None = all)')
    parser.add_argument('--no-limited-horizon', action='store_true',
                        help='Disable adaptive lookahead')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--optimize-hyperparams', action='store_true',
                        help='Run hyperparameter optimization')
    parser.add_argument('--optimization-dir', default='hyperparameter_optimization',
                        help='Directory for optimization results')
    parser.add_argument('--optimization-sample-size', type=int, default=10,
                        help='Number of instances to sample for optimization')
    parser.add_argument('--dataset', choices=['ca', 'ferc', 'rts_gmlc', 'all'], default='rts_gmlc',
                        help='Which dataset to benchmark')

    args = parser.parse_args()

    # Find all instances
    examples_path = Path(args.examples_dir)
    if not examples_path.exists():
        print(f"Error: Examples directory '{examples_path}' not found")
        return

    if args.dataset == 'all':
        instances = sorted(list(examples_path.glob('**/*.json')))
    else:
        instances = sorted(list(examples_path.glob(f'{args.dataset}/*.json')))

    if not instances:
        print(f"No instances found in {examples_path}")
        return

    print(f"Found {len(instances)} instances")

    # Run hyperparameter optimization if requested
    if args.optimize_hyperparams:
        optimize_hyperparameters(
            instances=instances,
            output_dir=args.optimization_dir,
            max_workers=args.workers,
            sample_size=args.optimization_sample_size
        )
    else:
        # Run benchmark with specified parameters
        run_benchmark(
            instances=instances,
            output_file=args.output,
            max_workers=args.workers,
            window_size=args.window_size,
            window_step=args.window_step,
            gap=0.01,
            generators_per_iteration=args.generators_per_iteration,
            use_limited_horizon=not args.no_limited_horizon,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main()
