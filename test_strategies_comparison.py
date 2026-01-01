"""
Comprehensive comparison of generator selection strategies for Relax-and-Fix
Includes baseline (direct solver) comparison
"""
import json
import time
from pyomo.opt import SolverFactory
from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix
from src.solvers.generator_selection_strategies import STRATEGIES, get_strategy

# Configuration
TEST_FILE = r"C:\Users\oQaris\Desktop\Git\uc\examples\ca\2015-03-01_reserves_3.json"
WINDOW_SIZE = 8
WINDOW_STEP = 8
GENERATORS_PER_ITERATION = 61  # None for classic R&F
GAP = 0.01
SOLVER_NAME = "appsi_highs"
BASELINE_TIME_LIMIT = 300  # 5 minutes for direct solver

print("="*80)
print("GENERATOR SELECTION STRATEGIES COMPARISON")
print("="*80)
print(f"Test instance: {TEST_FILE}")
print(f"Window size: {WINDOW_SIZE}, Step: {WINDOW_STEP}")
print(f"Generators per iteration: {GENERATORS_PER_ITERATION}")
print(f"MIP gap: {GAP}, Solver: {SOLVER_NAME}")
print("="*80)

# Load test data
with open(TEST_FILE, 'r') as f:
    data = json.load(f)

num_generators = len(data["thermal_generators"])
num_periods = data["time_periods"]
print(f"\nProblem size:")
print(f"  Thermal generators: {num_generators}")
print(f"  Time periods: {num_periods}")
print()

results = []

# =============================================================================
# BASELINE: Direct solver (no Relax-and-Fix)
# =============================================================================
print("="*80)
print("BASELINE: Direct Solver (No Relax-and-Fix)")
print("="*80)

model_baseline = build_uc_model(data)
solver = SolverFactory(SOLVER_NAME)

if hasattr(solver, 'config'):
    solver.config.mip_gap = GAP
    solver.config.time_limit = BASELINE_TIME_LIMIT

start_time = time.time()
result_baseline = solver.solve(model_baseline, tee=False)
baseline_time = time.time() - start_time

from pyomo.environ import value
baseline_obj = value(model_baseline.obj) if result_baseline.solver.termination_condition.name == 'optimal' else None
baseline_status = result_baseline.solver.termination_condition.name

print(f"  Time: {baseline_time:.2f}s")
print(f"  Objective: {baseline_obj:.2f}" if baseline_obj else f"  Status: {baseline_status}")
print(f"  Status: {baseline_status}")
print()

results.append({
    'method': 'BASELINE (Direct)',
    'strategy': '-',
    'time': baseline_time,
    'objective': baseline_obj if baseline_obj else float('inf'),
    'status': baseline_status,
    'feasible': baseline_status == 'optimal'
})

# =============================================================================
# RELAX-AND-FIX WITH DIFFERENT STRATEGIES
# =============================================================================
print("="*80)
print("RELAX-AND-FIX WITH DIFFERENT STRATEGIES")
print("="*80)
print()

# Test each strategy
for strategy_key, strategy_info in STRATEGIES.items():
    print(f"Testing: {strategy_info['name']} - {strategy_info['description']}")

    # Build fresh model
    model = build_uc_model(data)

    # Run Relax-and-Fix with this strategy
    try:
        result = solve_relax_and_fix(
            model=model,
            window_size=WINDOW_SIZE,
            window_step=WINDOW_STEP,
            gap=GAP,
            solver_name=SOLVER_NAME,
            verbose=False,
            verify_solution=False,  # Skip verification for speed
            data=data,
            model_builder=build_uc_model,
            generators_per_iteration=GENERATORS_PER_ITERATION,
            generator_sort_function=strategy_info['function']
        )

        print(f"  Time: {result['solve_time']:.2f}s")
        print(f"  Objective: {result['objective']:.2f}")
        print(f"  Status: {result['status']}")
        print()

        results.append({
            'method': 'Relax-and-Fix',
            'strategy': strategy_info['name'],
            'time': result['solve_time'],
            'objective': result['objective'],
            'status': result['status'],
            'feasible': result['status'] == 'completed'
        })

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        print()
        results.append({
            'method': 'Relax-and-Fix',
            'strategy': strategy_info['name'],
            'time': float('inf'),
            'objective': float('inf'),
            'status': f'error: {str(e)}',
            'feasible': False
        })

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("="*80)
print("SUMMARY REPORT")
print("="*80)
print()

# Sort by objective (best first)
results_sorted = sorted(results, key=lambda x: x['objective'])

print(f"{'Rank':<5} {'Method':<18} {'Strategy':<25} {'Time (s)':>10} {'Objective':>15} {'Status':<12}")
print("-"*95)

for i, res in enumerate(results_sorted, 1):
    method = res['method']
    strategy = res['strategy']
    time_val = f"{res['time']:.2f}" if res['time'] != float('inf') else "FAILED"
    obj_val = f"{res['objective']:.2f}" if res['objective'] != float('inf') else "INFEASIBLE"
    status = res['status']

    print(f"{i:<5} {method:<18} {strategy:<25} {time_val:>10} {obj_val:>15} {status:<12}")

print("="*80)

# Performance comparison with baseline
if baseline_obj is not None and baseline_obj != float('inf'):
    print("\nPERFORMANCE vs BASELINE:")
    print(f"{'Strategy':<25} {'Time Ratio':>12} {'Obj Gap %':>12} {'Speedup':>10}")
    print("-"*60)

    for res in results_sorted:
        if res['method'] == 'Relax-and-Fix' and res['feasible']:
            time_ratio = res['time'] / baseline_time if baseline_time > 0 else float('inf')
            obj_gap = ((res['objective'] - baseline_obj) / baseline_obj * 100) if baseline_obj > 0 else 0
            speedup = baseline_time / res['time'] if res['time'] > 0 else 0

            print(f"{res['strategy']:<25} {time_ratio:>12.2f}x {obj_gap:>11.2f}% {speedup:>9.2f}x")

    print("="*80)

# Best strategy recommendation
print("\nRECOMMENDATIONS:")
best_time = min([r for r in results if r['feasible']], key=lambda x: x['time'])
best_objective = min([r for r in results if r['feasible']], key=lambda x: x['objective'])

print(f"  Fastest: {best_time['strategy']} ({best_time['time']:.2f}s)")
print(f"  Best objective: {best_objective['strategy']} ({best_objective['objective']:.2f})")

if best_time == best_objective:
    print(f"  \u2605 WINNER: {best_time['strategy']} (best in both time AND objective)")
else:
    print(f"  Trade-off: Choose based on priority (speed vs quality)")

print("="*80)
