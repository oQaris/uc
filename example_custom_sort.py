"""
Example: Using custom generator sorting function
"""
import json
from src.models.uc_model import build_uc_model
from src.solvers import solve_relax_and_fix_v2

# Load data
with open(r"C:\Users\oQaris\Desktop\Git\uc\examples\ca\2015-03-01_reserves_3.json") as f:
    data = json.load(f)

# Build model
model = build_uc_model(data)


# Define custom sorting functions
def sort_by_startup_cost(data):
    """Sort generators by minimum startup cost (cheapest first)"""
    thermal_gens = data.get("thermal_generators", {})
    gen_cost_pairs = [
        (g, min(s.get("cost", float('inf')) for s in gen_data.get("startup", [{}])))
        for g, gen_data in thermal_gens.items()
    ]
    gen_cost_pairs.sort(key=lambda x: x[1])
    return [g for g, _ in gen_cost_pairs]


def sort_by_min_power(data):
    """Sort generators by minimum power output (smallest first)"""
    thermal_gens = data.get("thermal_generators", {})
    gen_power_pairs = [
        (g, gen_data.get("power_output_minimum", 0.0))
        for g, gen_data in thermal_gens.items()
    ]
    gen_power_pairs.sort(key=lambda x: x[1])
    return [g for g, _ in gen_power_pairs]


def sort_by_ramping_flexibility(data):
    """Sort generators by ramping capability (most flexible first)"""
    thermal_gens = data.get("thermal_generators", {})
    gen_ramp_pairs = [
        (g, gen_data.get("ramp_up_limit", 0.0) + gen_data.get("ramp_down_limit", 0.0))
        for g, gen_data in thermal_gens.items()
    ]
    gen_ramp_pairs.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in gen_ramp_pairs]


print("="*80)
print("CUSTOM GENERATOR SORTING EXAMPLES")
print("="*80)

# Test 1: Default sorting (by max power)
print("\n1. Default: Sort by max power (descending)")
result1 = solve_relax_and_fix_v2(
    model=build_uc_model(data),
    data=data,
    window_size=8,
    window_step=8,
    gap=0.01,
    verbose=False
)
print(f"   Objective: {result1['objective']:,.2f}")
print(f"   Time: {result1['solve_time']:.2f}s")

# Test 2: Sort by startup cost
print("\n2. Custom: Sort by startup cost (cheapest first)")
result2 = solve_relax_and_fix_v2(
    model=build_uc_model(data),
    data=data,
    window_size=8,
    window_step=8,
    gap=0.01,
    verbose=False,
    generator_sort_function=sort_by_startup_cost
)
print(f"   Objective: {result2['objective']:,.2f}")
print(f"   Time: {result2['solve_time']:.2f}s")

# Test 3: Sort by minimum power
print("\n3. Custom: Sort by min power (smallest first)")
result3 = solve_relax_and_fix_v2(
    model=build_uc_model(data),
    data=data,
    window_size=8,
    window_step=8,
    gap=0.01,
    verbose=False,
    generator_sort_function=sort_by_min_power
)
print(f"   Objective: {result3['objective']:,.2f}")
print(f"   Time: {result3['solve_time']:.2f}s")

# Test 4: Sort by ramping flexibility
print("\n4. Custom: Sort by ramping flexibility (most flexible first)")
result4 = solve_relax_and_fix_v2(
    model=build_uc_model(data),
    data=data,
    window_size=8,
    window_step=8,
    gap=0.01,
    verbose=False,
    generator_sort_function=sort_by_ramping_flexibility
)
print(f"   Objective: {result4['objective']:,.2f}")
print(f"   Time: {result4['solve_time']:.2f}s")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Default (max power):        {result1['objective']:,.2f}  ({result1['solve_time']:.2f}s)")
print(f"Startup cost:               {result2['objective']:,.2f}  ({result2['solve_time']:.2f}s)")
print(f"Min power:                  {result3['objective']:,.2f}  ({result3['solve_time']:.2f}s)")
print(f"Ramping flexibility:        {result4['objective']:,.2f}  ({result4['solve_time']:.2f}s)")
print()

# Find best
results = [
    ("Default (max power)", result1),
    ("Startup cost", result2),
    ("Min power", result3),
    ("Ramping flexibility", result4)
]
best = min(results, key=lambda x: x[1]['objective'])
print(f"Best strategy: {best[0]}")
print(f"  Objective: {best[1]['objective']:,.2f}")
print(f"  Time: {best[1]['solve_time']:.2f}s")
print("="*80)
