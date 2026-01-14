"""
Quick test to verify infeasibility handling in relax-and-fix
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix

# Use smaller RTS-GMLC instance for faster testing
DATA_FILE = r"C:\Users\oQaris\Desktop\Git\uc\examples\rts_gmlc\2020-07-06.json"
SOLVER = "appsi_highs"
GAP = 0.001
VERBOSE = True

print(f"Loading data from {DATA_FILE}")
with open(DATA_FILE) as f:
    data = json.load(f)

print(f"Instance: {data['time_periods']} periods, {len(data['thermal_generators'])} thermal gens")
print(f"Solver: {SOLVER}, Gap: {GAP}\n")

# Build model
model = build_uc_model(data)

# Test relax-and-fix with adaptive horizon
# Using small window/step to reach more windows faster
result = solve_relax_and_fix(
    model,
    window_size=8,
    window_step=8,
    solver_name=SOLVER,
    gap=GAP,
    verbose=VERBOSE,
    data=data,  # Required for generator sorting
    model_builder=build_uc_model,  # Required for solution verification
    use_limited_horizon=True,
    generators_per_iteration=16  # Fewer gens per iteration for faster testing
)

print("\n" + "=" * 60)
print("TEST COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Solve time: {result['solve_time']:.2f}s")
print(f"Objective: {result['objective']:.2f}")
print(f"Feasible: {result.get('feasible', 'N/A')}")
