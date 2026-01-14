"""
Quick example: Using optimized Relax-and-Fix v2
"""
import json
from src.models.uc_model import build_uc_model
from src.solvers import solve_relax_and_fix_v2  # Clean import!

# Load data
with open(r"C:\Users\oQaris\Desktop\Git\uc\examples\ca\2015-03-01_reserves_3.json") as f:
    data = json.load(f)

# Build model
model = build_uc_model(data)

# Solve with optimized R&F
print("Solving with optimized Relax-and-Fix v2...")
result = solve_relax_and_fix_v2(
    model=model,
    data=data,
    window_size=8,
    window_step=8,
    gap=0.01,
    verbose=True,
    generators_per_iteration=61
)

print("\n" + "="*60)
print("RESULT")
print("="*60)
print(f"Objective:  {result['objective']:,.2f}")
print(f"Time:       {result['solve_time']:.2f}s")
print(f"Status:     {result['status']}")
print("="*60)
