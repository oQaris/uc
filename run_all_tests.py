#!/usr/bin/env python3
"""
Run UC model on all test instances from pglib-uc repository and log solve times
"""
import os
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import csv

# Import pyomo modules
from pyomo.environ import *
from pyomo.opt import SolverFactory


def load_and_solve_instance(data_file: str, solver_name: str = 'cbc', ratio_gap: float = 0.01,
                            time_limit: int = None, verbose: bool = False):
    """
    Load a UC instance from JSON file and solve it

    Returns:
        dict: Results including solve time, objective value, and solver status
    """
    results = {
        'instance': os.path.basename(data_file),
        'file_path': data_file,
        'status': 'unknown',
        'solve_time': None,
        'objective_value': None,
        'gap': None,
        'error': None
    }

    try:
        # Load data
        print(f"\nLoading data from {data_file}")
        start_load = time.time()
        with open(data_file, 'r') as f:
            data = json.load(f)
        load_time = time.time() - start_load
        print(f"  Data loaded in {load_time:.2f}s")

        thermal_gens = data['thermal_generators']
        renewable_gens = data['renewable_generators']

        time_periods = {t+1 : t for t in range(data['time_periods'])}

        gen_startup_categories = {g : list(range(0, len(gen['startup']))) for (g, gen) in thermal_gens.items()}
        gen_pwl_points = {g : list(range(0, len(gen['piecewise_production']))) for (g, gen) in thermal_gens.items()}

        # Build model
        print("  Building model...")
        start_build = time.time()
        m = ConcreteModel()

        # Variables
        m.cg = Var(thermal_gens.keys(), time_periods.keys())
        m.pg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)
        m.rg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)
        m.pw = Var(renewable_gens.keys(), time_periods.keys(), within=NonNegativeReals)
        m.ug = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)
        m.vg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)
        m.wg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)

        m.dg = Var(((g,s,t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods), within=Binary)
        m.lg = Var(((g,l,t) for g in thermal_gens for l in gen_pwl_points[g] for t in time_periods), within=UnitInterval)

        # Objective
        m.obj = Objective(expr=sum(
                                  sum(
                                      m.cg[g,t] + gen['piecewise_production'][0]['cost']*m.ug[g,t]
                                      + sum( gen_startup['cost']*m.dg[g,s,t] for (s, gen_startup) in enumerate(gen['startup']))
                                  for t in time_periods)
                                for g, gen in thermal_gens.items() )
                                )

        # System-wide constraints
        m.demand = Constraint(time_periods.keys())
        m.reserves = Constraint(time_periods.keys())
        for t,t_idx in time_periods.items():
            m.demand[t] = sum( m.pg[g,t]+gen['power_output_minimum']*m.ug[g,t] for (g, gen) in thermal_gens.items() ) + sum( m.pw[w,t] for w in renewable_gens ) == data['demand'][t_idx]
            m.reserves[t] = sum( m.rg[g,t] for g in thermal_gens ) >= data['reserves'][t_idx]

        # Initial time constraints
        m.uptimet0 = Constraint(thermal_gens.keys())
        m.downtimet0 = Constraint(thermal_gens.keys())
        m.logicalt0 = Constraint(thermal_gens.keys())
        m.startupt0 = Constraint(thermal_gens.keys())
        m.rampupt0 = Constraint(thermal_gens.keys())
        m.rampdownt0 = Constraint(thermal_gens.keys())
        m.shutdownt0 = Constraint(thermal_gens.keys())

        for g, gen in thermal_gens.items():
            if gen['unit_on_t0'] == 1:
                if gen['time_up_minimum'] - gen['time_up_t0'] >= 1:
                    m.uptimet0[g] = sum( (m.ug[g,t] - 1) for t in range(1, min(gen['time_up_minimum'] - gen['time_up_t0'], data['time_periods'])+1)) == 0
            elif gen['unit_on_t0'] == 0:
                if gen['time_down_minimum'] - gen['time_down_t0'] >= 1:
                    m.downtimet0[g] = sum( m.ug[g,t] for t in range(1, min(gen['time_down_minimum'] - gen['time_down_t0'], data['time_periods'])+1)) == 0
            else:
                raise Exception('Invalid unit_on_t0 for generator {}, unit_on_t0={}'.format(g, gen['unit_on_t0']))

            m.logicalt0[g] = m.ug[g,1] - gen['unit_on_t0'] == m.vg[g,1] - m.wg[g,1]

            startup_expr = sum(
                                sum( m.dg[g,s,t]
                                        for t in range(
                                                        max(1,gen['startup'][s+1]['lag']-gen['time_down_t0']+1),
                                                        min(gen['startup'][s+1]['lag']-1,data['time_periods'])+1
                                                      )
                                    )
                               for s,_ in enumerate(gen['startup'][:-1]))
            if isinstance(startup_expr, int):
                pass
            else:
                m.startupt0[g] = startup_expr == 0

            m.rampupt0[g] = m.pg[g,1] + m.rg[g,1] - gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['ramp_up_limit']
            m.rampdownt0[g] = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) - m.pg[g,1] <= gen['ramp_down_limit']

            shutdown_constr = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['unit_on_t0']*(gen['power_output_maximum'] - gen['power_output_minimum']) - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,1]

            if isinstance(shutdown_constr, bool):
                pass
            else:
                m.shutdownt0[g] = shutdown_constr

        # Generator constraints
        m.mustrun = Constraint(thermal_gens.keys(), time_periods.keys())
        m.logical = Constraint(thermal_gens.keys(), time_periods.keys())
        m.uptime = Constraint(thermal_gens.keys(), time_periods.keys())
        m.downtime = Constraint(thermal_gens.keys(), time_periods.keys())
        m.startup_select = Constraint(thermal_gens.keys(), time_periods.keys())
        m.gen_limit1 = Constraint(thermal_gens.keys(), time_periods.keys())
        m.gen_limit2 = Constraint(thermal_gens.keys(), time_periods.keys())
        m.ramp_up = Constraint(thermal_gens.keys(), time_periods.keys())
        m.ramp_down = Constraint(thermal_gens.keys(), time_periods.keys())
        m.power_select = Constraint(thermal_gens.keys(), time_periods.keys())
        m.cost_select = Constraint(thermal_gens.keys(), time_periods.keys())
        m.on_select = Constraint(thermal_gens.keys(), time_periods.keys())

        for g, gen in thermal_gens.items():
            for t in time_periods:
                m.mustrun[g,t] = m.ug[g,t] >= gen['must_run']

                if t > 1:
                    m.logical[g,t] = m.ug[g,t] - m.ug[g,t-1] == m.vg[g,t] - m.wg[g,t]

                UT = min(gen['time_up_minimum'],data['time_periods'])
                if t >= UT:
                    m.uptime[g,t] = sum(m.vg[g,t] for t in range(t-UT+1, t+1)) <= m.ug[g,t]
                DT = min(gen['time_down_minimum'],data['time_periods'])
                if t >= DT:
                    m.downtime[g,t] = sum(m.wg[g,t] for t in range(t-DT+1, t+1)) <= 1-m.ug[g,t]
                m.startup_select[g,t] = m.vg[g,t] == sum(m.dg[g,s,t] for s,_ in enumerate(gen['startup']))

                m.gen_limit1[g,t] = m.pg[g,t]+m.rg[g,t] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_startup_limit']),0)*m.vg[g,t]

                if t < len(time_periods):
                    m.gen_limit2[g,t] = m.pg[g,t]+m.rg[g,t] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,t+1]

                if t > 1:
                    m.ramp_up[g,t] = m.pg[g,t]+m.rg[g,t] - m.pg[g,t-1] <= gen['ramp_up_limit']
                    m.ramp_down[g,t] = m.pg[g,t-1] - m.pg[g,t] <= gen['ramp_down_limit']

                piece_mw1 = gen['piecewise_production'][0]['mw']
                piece_cost1 = gen['piecewise_production'][0]['cost']
                m.power_select[g,t] = m.pg[g,t] == sum( (piece['mw'] - piece_mw1)*m.lg[g,l,t] for l,piece in enumerate(gen['piecewise_production']))
                m.cost_select[g,t] = m.cg[g,t] == sum( (piece['cost'] - piece_cost1)*m.lg[g,l,t] for l,piece in enumerate(gen['piecewise_production']))
                m.on_select[g,t] = m.ug[g,t] == sum(m.lg[g,l,t] for l,_ in enumerate(gen['piecewise_production']))

        m.startup_allowed = Constraint(((g,s,t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods))
        for g, gen in thermal_gens.items():
            for s,_ in enumerate(gen['startup'][:-1]):
                for t in time_periods:
                    if t >= gen['startup'][s+1]['lag']:
                        m.startup_allowed[g,s,t] = m.dg[g,s,t] <= sum(m.wg[g,t-i] for i in range(gen['startup'][s]['lag'], gen['startup'][s+1]['lag']))

        # Renewable constraints
        for w, gen in renewable_gens.items():
            for t, t_idx in time_periods.items():
                m.pw[w,t].setlb(gen['power_output_minimum'][t_idx])
                m.pw[w,t].setub(gen['power_output_maximum'][t_idx])

        build_time = time.time() - start_build
        print(f"  Model built in {build_time:.2f}s")

        # Solve
        print("  Solving...")
        solver = SolverFactory(solver_name)

        solver_options = {'ratioGap': ratio_gap}
        if time_limit:
            solver_options['seconds'] = time_limit

        start_solve = time.time()
        solve_result = solver.solve(m, options=solver_options, tee=verbose)
        solve_time = time.time() - start_solve

        # Extract results
        results['solve_time'] = solve_time
        results['build_time'] = build_time
        results['load_time'] = load_time
        results['total_time'] = load_time + build_time + solve_time

        # Check solver status
        if (solve_result.solver.status == SolverStatus.ok):
            if (solve_result.solver.termination_condition == TerminationCondition.optimal):
                results['status'] = 'optimal'
            elif (solve_result.solver.termination_condition == TerminationCondition.feasible):
                results['status'] = 'feasible'
            else:
                results['status'] = str(solve_result.solver.termination_condition)
        else:
            results['status'] = str(solve_result.solver.status)

        # Get objective value if available
        try:
            results['objective_value'] = value(m.obj)
        except:
            results['objective_value'] = None

        # Try to get gap
        try:
            if hasattr(solve_result.problem, 'upper_bound') and hasattr(solve_result.problem, 'lower_bound'):
                ub = solve_result.problem.upper_bound
                lb = solve_result.problem.lower_bound
                if ub and lb and lb != 0:
                    results['gap'] = abs((ub - lb) / lb) * 100
        except:
            pass

        print(f"  Status: {results['status']}")
        print(f"  Solve time: {solve_time:.2f}s")
        if results['objective_value']:
            print(f"  Objective: {results['objective_value']:.2f}")

    except Exception as e:
        results['error'] = str(e)
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    return results


def find_all_test_instances(base_dir: str = '.'):
    """Find all JSON test instances in the repository"""
    test_dirs = ['ca', 'ferc', 'rts_gmlc']
    instances = []

    for test_dir in test_dirs:
        dir_path = os.path.join(base_dir, test_dir)
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith('.json'):
                    instances.append(os.path.join(dir_path, file))

    return sorted(instances)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run UC model on all test instances')
    parser.add_argument('--solver', type=str, default='cbc', help='Solver to use (default: cbc)')
    parser.add_argument('--gap', type=float, default=0.01, help='MIP gap tolerance (default: 0.01)')
    parser.add_argument('--time-limit', type=int, default=None, help='Time limit per instance in seconds')
    parser.add_argument('--output', type=str, default='test_results.csv', help='Output CSV file')
    parser.add_argument('--verbose', action='store_true', help='Show solver output')
    parser.add_argument('--instances', nargs='+', help='Specific instances to run (default: all)')
    parser.add_argument('--limit', type=int, help='Limit number of instances to run')

    args = parser.parse_args()

    # Find test instances
    if args.instances:
        instances = args.instances
    else:
        instances = find_all_test_instances()

    if args.limit:
        instances = instances[:args.limit]

    print(f"Found {len(instances)} test instances")
    print(f"Solver: {args.solver}")
    print(f"MIP gap: {args.gap}")
    if args.time_limit:
        print(f"Time limit: {args.time_limit}s per instance")
    print()

    # Initialize CSV file with header
    fieldnames = ['instance', 'status', 'solve_time', 'build_time', 'load_time',
                 'total_time', 'objective_value', 'gap', 'error', 'file_path']

    # Write CSV header
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Run all instances
    all_results = []
    start_time = time.time()

    for i, instance in enumerate(instances, 1):
        print(f"\n{'='*80}")
        print(f"Instance {i}/{len(instances)}: {instance}")
        print(f"{'='*80}")

        result = load_and_solve_instance(
            instance,
            solver_name=args.solver,
            ratio_gap=args.gap,
            time_limit=args.time_limit,
            verbose=args.verbose
        )

        all_results.append(result)

        # Write result immediately to CSV (append mode)
        with open(args.output, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)

        print(f"  >> Result saved to {args.output}")

    total_time = time.time() - start_time

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\nResults saved to {args.output}")
    print(f"\nTotal instances: {len(instances)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    # Print summary statistics
    successful = [r for r in all_results if r['status'] in ['optimal', 'feasible']]
    failed = [r for r in all_results if r['status'] not in ['optimal', 'feasible']]

    print(f"\nSuccessful solves: {len(successful)}")
    print(f"Failed solves: {len(failed)}")

    if successful:
        solve_times = [r['solve_time'] for r in successful if r['solve_time']]
        if solve_times:
            print(f"\nSolve time statistics (successful instances):")
            print(f"  Min: {min(solve_times):.2f}s")
            print(f"  Max: {max(solve_times):.2f}s")
            print(f"  Avg: {sum(solve_times)/len(solve_times):.2f}s")
            print(f"  Total: {sum(solve_times):.2f}s")

    if failed:
        print(f"\nFailed instances:")
        for r in failed:
            print(f"  {r['instance']}: {r['status']} - {r['error'] if r['error'] else 'N/A'}")


if __name__ == "__main__":
    main()
