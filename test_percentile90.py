"""
Quick test for adaptive lookahead strategy with window_size=16
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix

DATA_FILE = r"examples\rts_gmlc\2020-07-06.json"
SOLVER = "appsi_highs"
GAP = 0.001
VERBOSE = True

print(f"Loading data from {DATA_FILE}")
with open(DATA_FILE) as f:
    data = json.load(f)

print(f"Instance: {data['time_periods']} periods, {len(data['thermal_generators'])} thermal gens")
print(f"Solver: {SOLVER}, Gap: {GAP}")

print("\n" + "=" * 60)
print("Testing ADAPTIVE lookahead strategy (window_size=16)")
print("=" * 60)

model = build_uc_model(data)

result = solve_relax_and_fix(
    model, window_size=16, window_step=16,
    solver_name=SOLVER, gap=GAP, verbose=VERBOSE,
    data=data, model_builder=build_uc_model,
    use_limited_horizon=True
)

print(f"\nSolve time: {result['solve_time']:.2f}s")
print(f"Objective: {result['objective']:.2f}")

if 'feasible' in result:
    feasible_str = "FEASIBLE" if result['feasible'] else "INFEASIBLE"
    print(f"Verification: {feasible_str}")
    if result['feasible']:
        print(f"  Verified objective: {result['verified_objective']:.2f}")
        print(f"  Gap: {result['objective_gap']:.2f}")