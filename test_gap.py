"""Test if mip_gap works correctly for appsi_highs"""

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.uc_model import build_uc_model
from pyomo.opt import SolverFactory
from pyomo.environ import value

# Load instance
instance_path = r'examples\rts_gmlc\2020-07-06.json'
with open(instance_path, 'r') as f:
    data = json.load(f)

print("Building model...")
model = build_uc_model(data)

# Test: config.mip_gap vs highs_options['mip_rel_gap']
print("\nTesting mip_gap configuration...")
solver = SolverFactory("appsi_highs")

# Check what mip_gap options are available
print("\nTesting solver.config.mip_gap...")
solver.config.mip_gap = 0.05  # 5% gap
print(f"Set solver.config.mip_gap = {solver.config.mip_gap}")

start = time.time()
result1 = solver.solve(model, tee=False)
time1 = time.time() - start

print(f"Time with config.mip_gap=0.05: {time1:.2f}s")
print(f"Status: {result1.solver.termination_condition}")

# Now test with highs_options
print("\nTesting solver.highs_options['mip_rel_gap']...")
solver2 = SolverFactory("appsi_highs")
solver2.highs_options['mip_rel_gap'] = 0.05
print(f"Set solver.highs_options['mip_rel_gap'] = 0.05")

# Rebuild model
model2 = build_uc_model(data)

start = time.time()
result2 = solver2.solve(model2, tee=False)
time2 = time.time() - start

print(f"Time with highs_options['mip_rel_gap']=0.05: {time2:.2f}s")
print(f"Status: {result2.solver.termination_condition}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Method 1 (config.mip_gap):           {time1:.2f}s")
print(f"Method 2 (highs_options['mip_rel_gap']): {time2:.2f}s")
print("\nBoth methods should give similar times if working correctly.")
