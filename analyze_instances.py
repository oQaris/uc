#!/usr/bin/env python3
"""
Analyze UC instances to extract metadata about complexity
"""
import json
import os
from pathlib import Path

def analyze_instance(file_path):
    """Extract metadata from a UC instance"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    thermal_gens = data['thermal_generators']
    renewable_gens = data['renewable_generators']
    time_periods = data['time_periods']

    metadata = {
        'instance': os.path.basename(file_path),
        'file_path': file_path,
        'time_periods': time_periods,
        'n_thermal_gens': len(thermal_gens),
        'n_renewable_gens': len(renewable_gens),
        'total_generators': len(thermal_gens) + len(renewable_gens),
    }

    # Analyze thermal generators
    must_run_count = sum(1 for g in thermal_gens.values() if g['must_run'] == 1)
    metadata['n_must_run'] = must_run_count

    # Count startup categories
    total_startup_categories = sum(len(g['startup']) for g in thermal_gens.values())
    metadata['total_startup_categories'] = total_startup_categories
    metadata['avg_startup_categories'] = total_startup_categories / len(thermal_gens) if thermal_gens else 0

    # Count piecewise points
    total_pwl_points = sum(len(g['piecewise_production']) for g in thermal_gens.values())
    metadata['total_pwl_points'] = total_pwl_points
    metadata['avg_pwl_points'] = total_pwl_points / len(thermal_gens) if thermal_gens else 0

    # Capacity statistics
    total_thermal_capacity = sum(g['power_output_maximum'] for g in thermal_gens.values())
    metadata['total_thermal_capacity'] = round(total_thermal_capacity, 2)

    # Demand statistics
    total_demand = sum(data['demand'])
    peak_demand = max(data['demand'])
    avg_demand = total_demand / len(data['demand'])
    metadata['total_demand'] = round(total_demand, 2)
    metadata['peak_demand'] = round(peak_demand, 2)
    metadata['avg_demand'] = round(avg_demand, 2)

    # Reserve requirements
    total_reserves = sum(data['reserves'])
    metadata['total_reserves'] = round(total_reserves, 2)
    metadata['avg_reserves'] = round(total_reserves / len(data['reserves']), 2)

    # Calculate approximate problem size
    # Variables: cg, pg, rg (thermal), pw (renewable), ug, vg, wg (binary thermal)
    # + dg (startup categories * thermal * time), lg (pwl points * thermal * time)
    n_continuous_vars = len(thermal_gens) * time_periods * 3  # cg, pg, rg
    n_continuous_vars += len(renewable_gens) * time_periods  # pw
    n_binary_vars = len(thermal_gens) * time_periods * 3  # ug, vg, wg
    n_binary_vars += total_startup_categories * time_periods  # dg
    n_continuous_vars += total_pwl_points * time_periods  # lg

    metadata['approx_continuous_vars'] = n_continuous_vars
    metadata['approx_binary_vars'] = n_binary_vars
    metadata['approx_total_vars'] = n_continuous_vars + n_binary_vars

    # Estimate number of constraints (rough approximation)
    # Demand + reserves per time period
    n_constraints = time_periods * 2
    # Generator constraints (multiple types per generator per time)
    n_constraints += len(thermal_gens) * time_periods * 10  # rough estimate
    # Startup constraints
    n_constraints += total_startup_categories * time_periods
    # Initial time constraints
    n_constraints += len(thermal_gens) * 7

    metadata['approx_constraints'] = n_constraints

    # Calculate "complexity score" - weighted combination of factors
    # Higher score = more complex
    complexity_score = (
        n_binary_vars * 10 +  # Binary vars are most expensive
        n_continuous_vars * 1 +
        n_constraints * 0.1 +
        total_startup_categories * 100 +  # Startup logic is complex
        must_run_count * 50  # Must-run constraints reduce flexibility
    )
    metadata['complexity_score'] = int(complexity_score)

    return metadata

if __name__ == "__main__":
    # Test on a few instances
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        metadata = analyze_instance(file_path)
        for key, value in metadata.items():
            print(f"{key}: {value}")
    else:
        # Analyze all instances
        from run_all_tests_parallel import find_all_test_instances
        instances = find_all_test_instances()

        print("Instance analysis:")
        print(f"{'Instance':<40} {'Gens':<6} {'Time':<5} {'Binaries':<8} {'Complexity':<12}")
        print("-" * 80)

        for instance in instances:  # First 10 for testing
            meta = analyze_instance(instance)
            print(f"{meta['instance']:<40} {meta['total_generators']:<6} "
                  f"{meta['time_periods']:<5} {meta['approx_binary_vars']:<8} "
                  f"{meta['complexity_score']:<12}")
