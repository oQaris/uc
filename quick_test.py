"""Test benchmark_relax_and_fix with corrected time_limit"""

from pathlib import Path
import json
from src.runners.benchmark_relax_and_fix import benchmark_single_instance

# Test on a single instance
example = "examples/rts_gmlc/2020-07-06.json"

print("Testing fixed benchmark_single_instance on:", example)
print("="*60)

result = benchmark_single_instance(
    instance_path=example,
    window_size=8,
    window_step=4,
    gap=0.01,
    generators_per_iteration=None,
    use_limited_horizon=True,
    verbose=True
)

if result:
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"R&F time:    {result['rf_time']:.2f}s")
    print(f"HiGHS time:  {result['highs_time']:.2f}s (with {result['rf_time']:.2f}s limit)")
    print(f"HiGHS status: {result['highs_status']}")

    # Check if time limit worked
    if result['highs_time'] <= result['rf_time'] * 2:
        print("\nOK - HiGHS time limit appears to be working!")
    else:
        print(f"\nWARNING - HiGHS took {result['highs_time']:.2f}s despite {result['rf_time']:.2f}s limit")
else:
    print("\nTest failed")
