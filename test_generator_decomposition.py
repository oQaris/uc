"""
Test script for Relax-and-Fix with generator decomposition
"""
import json
import time
from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix

# Load test instance (small RTS-GMLC instance)
test_file = "examples/rts_gmlc/2020-01-27.json"
print(f"Loading test instance: {test_file}")

with open(test_file, 'r') as f:
    data = json.load(f)

num_generators = len(data["thermal_generators"])
print(f"Number of thermal generators: {num_generators}")
print(f"Time periods: {data['time_periods']}")

# Test 1: Original Relax-and-Fix (no generator decomposition)
print("\n" + "="*80)
print("Test 1: Original Relax-and-Fix (all generators at once)")
print("="*80)

model1 = build_uc_model(data)
result1 = solve_relax_and_fix(
    model=model1,
    window_size=8,
    window_step=4,
    gap=0.01,
    solver_name="appsi_highs",
    verbose=True,
    verify_solution=True,
    data=data,
    model_builder=build_uc_model,
    generators_per_iteration=None  # No generator decomposition
)

print(f"\nResult 1:")
print(f"  Solve time: {result1['solve_time']:.2f}s")
print(f"  Objective: {result1['objective']:.2f}")
print(f"  Status: {result1['status']}")
if 'verification' in result1:
    print(f"  Verified feasible: {result1['verification']['feasible']}")
    if result1['verification']['feasible']:
        print(f"  Verified objective: {result1['verified_objective']:.2f}")
        print(f"  Objective gap: {result1['objective_gap']:.2f}")

# Test 2: Relax-and-Fix with generator decomposition (10 generators per iteration)
print("\n" + "="*80)
print("Test 2: Relax-and-Fix with generator decomposition (10 generators per iteration)")
print("="*80)

model2 = build_uc_model(data)
result2 = solve_relax_and_fix(
    model=model2,
    window_size=8,
    window_step=4,
    gap=0.01,
    solver_name="appsi_highs",
    verbose=True,
    verify_solution=True,
    data=data,
    model_builder=build_uc_model,
    generators_per_iteration=10  # Process 10 generators at a time
)

print(f"\nResult 2:")
print(f"  Solve time: {result2['solve_time']:.2f}s")
print(f"  Objective: {result2['objective']:.2f}")
print(f"  Status: {result2['status']}")
if 'verification' in result2:
    print(f"  Verified feasible: {result2['verification']['feasible']}")
    if result2['verification']['feasible']:
        print(f"  Verified objective: {result2['verified_objective']:.2f}")
        print(f"  Objective gap: {result2['objective_gap']:.2f}")

# Test 3: Relax-and-Fix with smaller batches (5 generators per iteration)
print("\n" + "="*80)
print("Test 3: Relax-and-Fix with generator decomposition (5 generators per iteration)")
print("="*80)

model3 = build_uc_model(data)
result3 = solve_relax_and_fix(
    model=model3,
    window_size=8,
    window_step=4,
    gap=0.01,
    solver_name="appsi_highs",
    verbose=True,
    verify_solution=True,
    data=data,
    model_builder=build_uc_model,
    generators_per_iteration=5  # Process 5 generators at a time
)

print(f"\nResult 3:")
print(f"  Solve time: {result3['solve_time']:.2f}s")
print(f"  Objective: {result3['objective']:.2f}")
print(f"  Status: {result3['status']}")
if 'verification' in result3:
    print(f"  Verified feasible: {result3['verification']['feasible']}")
    if result3['verification']['feasible']:
        print(f"  Verified objective: {result3['verified_objective']:.2f}")
        print(f"  Objective gap: {result3['objective_gap']:.2f}")

# Summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print(f"{'Method':<50} {'Time (s)':>12} {'Objective':>15} {'Verified':>10}")
print("-"*80)
print(f"{'Original (all generators)':50} {result1['solve_time']:>12.2f} {result1['objective']:>15.2f} {str(result1.get('feasible', 'N/A')):>10}")
print(f"{'With decomposition (10 gens/iter)':50} {result2['solve_time']:>12.2f} {result2['objective']:>15.2f} {str(result2.get('feasible', 'N/A')):>10}")
print(f"{'With decomposition (5 gens/iter)':50} {result3['solve_time']:>12.2f} {result3['objective']:>15.2f} {str(result3.get('feasible', 'N/A')):>10}")
print("="*80)
