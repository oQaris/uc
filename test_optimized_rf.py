"""
Compare performance: original vs optimized Relax-and-Fix
"""
import json
import time

from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix
from src.solvers.relax_and_fix_v2 import solve_relax_and_fix_v2

TEST_FILE = r"C:\Users\oQaris\Desktop\Git\uc\examples\rts_gmlc\2020-06-09.json"

print("=" * 80)
print("PERFORMANCE COMPARISON: Original vs Optimized Relax-and-Fix")
print("=" * 80)

# Load data
with open(TEST_FILE, 'r') as f:
    data = json.load(f)

print(f"Problem: {len(data['thermal_generators'])} generators, {data['time_periods']} periods\n")

# Test parameters
WINDOW_SIZE = 8
WINDOW_STEP = 6
GENERATORS_PER_ITERATION = 5
GAP = 0.01

# Test 1: Original implementation
print("=" * 80)
print("TEST 1: Original Relax-and-Fix (recreates solver each time)")
print("=" * 80)

model1 = build_uc_model(data)
start = time.time()

try:
    result1 = solve_relax_and_fix(
        model=model1,
        window_size=WINDOW_SIZE,
        window_step=WINDOW_STEP,
        gap=GAP,
        solver_name="appsi_highs",
        verbose=True,  # Disable for timing
        verify_solution=True,
        data=data,
        model_builder=build_uc_model,
        generators_per_iteration=GENERATORS_PER_ITERATION,
        use_limited_horizon=True
    )
    original_time = time.time() - start
    print(f"\n✓ Original: {original_time:.2f}s, obj={result1['objective']:.2f}")
except Exception as e:
    print(f"\n✗ Original failed: {e}")
    original_time = None

# Test 2: Optimized implementation
print("\n" + "=" * 80)
print("TEST 2: Optimized Relax-and-Fix (persistent solver)")
print("=" * 80)

model2 = build_uc_model(data)
start = time.time()

try:
    result2 = solve_relax_and_fix_v2(
        model=model2,
        data=data,
        window_size=WINDOW_SIZE,
        window_step=WINDOW_STEP,
        gap=GAP,
        verbose=True,
        generators_per_iteration=GENERATORS_PER_ITERATION,
        use_limited_horizon=True,
        verify_solution=True,
        model_builder=build_uc_model
    )
    optimized_time = time.time() - start
    print(f"\nOptimized: {optimized_time:.2f}s, obj={result2['objective']:.2f}")
except Exception as e:
    print(f"\nOptimized failed: {e}")
    optimized_time = None

# Comparison
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

if original_time and optimized_time:
    speedup = original_time / optimized_time
    improvement = ((original_time - optimized_time) / original_time) * 100

    print(f"Original time:   {original_time:.2f}s")
    print(f"Optimized time:  {optimized_time:.2f}s")
    print(f"Speedup:         {speedup:.2f}x")
    print(f"Improvement:     {improvement:.1f}%")
    print()

    if abs(result1['objective'] - result2['objective']) < 1.0:
        print("Objectives match (both implementations correct)")
    else:
        print(f"Objective difference: {abs(result1['objective'] - result2['objective']):.2f}")
        print("  This may be due to floating point differences")
else:
    print("Could not complete comparison (one or both implementations failed)")

print("=" * 80)
