"""
Compare direct solve vs Relax-and-Fix approach
"""
import json
import os
import sys
import time
from pathlib import Path

from pyomo.environ import value
from pyomo.opt import SolverFactory

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add conda bin to PATH for CBC solver
conda_bin = Path.home() / "miniconda3" / "Library" / "bin"
if conda_bin.exists():
    os.environ['PATH'] = str(conda_bin) + os.pathsep + os.environ.get('PATH', '')

from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix


def solve_direct(data, solver_name="cbc", gap=0.01, verbose=False):
    """Solve UC problem directly"""
    print("\n" + "=" * 60)
    print(f"DIRECT SOLVE ({solver_name.upper()})")
    print("=" * 60)

    start = time.time()
    model = build_uc_model(data)
    build_time = time.time() - start
    print(f"Build time: {build_time:.2f}s")

    solver = SolverFactory(solver_name)

    # Set gap depending on solver type
    if hasattr(solver, 'config'):
        solver.config.mip_gap = gap
        solver.solve(model)
    else:
        solver.solve(model, options={'ratioGap': gap})

    solve_time = time.time() - start - build_time

    obj = value(model.obj)

    print(f"Solve time: {solve_time:.2f}s")
    print(f"Objective: {obj:.2f}")

    return {
        'build_time': build_time,
        'solve_time': solve_time,
        'total_time': build_time + solve_time,
        'objective': obj
    }


def solve_rf(data, window_size, window_step, solver_name="cbc", gap=0.01, verbose=False, use_limited_horizon=False):
    """Solve UC problem with Relax-and-Fix"""
    print("\n" + "=" * 60)
    print(f"RELAX-AND-FIX {solver_name.upper()} (window={window_size}, step={window_step})")
    print("=" * 60)

    start = time.time()
    model = build_uc_model(data)
    build_time = time.time() - start
    print(f"Build time: {build_time:.2f}s")

    result = solve_relax_and_fix(
        model, window_size, window_step,
        solver_name=solver_name, gap=gap, verbose=verbose,
        data=data, model_builder=build_uc_model,
        use_limited_horizon=use_limited_horizon
    )

    print(f"Solve time: {result['solve_time']:.2f}s")
    print(f"Objective: {result['objective']:.2f}")

    if 'feasible' in result:
        feasible_str = "✓ FEASIBLE" if result['feasible'] else "✗ INFEASIBLE"
        print(f"Verification: {feasible_str}")
        if result['feasible']:
            print(f"  Verified objective: {result['verified_objective']:.2f}")
            print(f"  Gap: {result['objective_gap']:.2f}")

    return {
        'build_time': build_time,
        'solve_time': result['solve_time'],
        'total_time': build_time + result['solve_time'],
        'objective': result['objective'],
        'feasible': result.get('feasible', None),
        'verified_objective': result.get('verified_objective', None)
    }


def main():
    # === CONFIGURATION ===
    SOLVER = "appsi_highs"  # Options: "cbc", "appsi_highs"
    GAP = 0.001
    DATA_FILE = r"..\..\examples\rts_gmlc\2020-07-06.json"
    VERBOSE = True
    # ====================

    print(f"Loading data from {DATA_FILE}")
    with open(DATA_FILE) as f:
        data = json.load(f)

    print(f"Instance: {data['time_periods']} periods, {len(data['thermal_generators'])} thermal gens")
    print(f"Solver: {SOLVER}, Gap: {GAP}")

    # Direct solve
    direct = solve_direct(data, solver_name=SOLVER, gap=GAP, verbose=VERBOSE)

    # Relax-and-Fix with different window sizes
    # rf_8_8 = solve_rf(data, window_size=8, window_step=8, solver_name=SOLVER, gap=GAP, verbose=VERBOSE)
    rf_8_6 = solve_rf(data, window_size=8, window_step=8, solver_name=SOLVER, gap=GAP, verbose=VERBOSE, use_limited_horizon=True)
    # rf_6_4 = solve_rf(data, window_size=8, window_step=8, solver_name=SOLVER, gap=GAP, verbose=VERBOSE, use_limited_horizon=True, lookahead_strategy="percentile90")
    # rf_6_4 = solve_rf(data, window_size=6, window_step=4, solver_name=SOLVER, gap=GAP, verbose=VERBOSE)
    # rf_12_10 = solve_rf(data, window_size=12, window_step=10, solver_name=SOLVER, gap=GAP, verbose=VERBOSE)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<30} {'Time (s)':<12} {'Objective':<15} {'Gap (%)':<10} {'Speedup':<10}")
    print("-" * 80)

    def print_row(name, res, base_time, base_obj):
        gap = 100 * (res['objective'] - base_obj) / base_obj
        speedup = base_time / res['total_time']
        print(f"{name:<30} {res['total_time']:<12.2f} {res['objective']:<15.2f} {gap:<10.3f} {speedup:<10.2f}x")

    print_row("Direct", direct, direct['total_time'], direct['objective'])
    # print_row("R&F (no_horizon)", rf_8_8, direct['total_time'], direct['objective'])
    print_row("R&F (zero)", rf_8_6, direct['total_time'], direct['objective'])
    # print_row("R&F (percentile75)", rf_6_4, direct['total_time'], direct['objective'])
    # print_row("R&F (window=12, step=10)", rf_12_10, direct['total_time'], direct['objective'])


#

if __name__ == "__main__":
    main()
