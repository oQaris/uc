#!/usr/bin/env python3
"""
Run UC model on all test instances from pglib-uc repository with parallel execution
Supports HiGHS solver with multithreading
Features:
- Resume from where it stopped (skip already solved instances)
- Instance metadata for complexity analysis
"""
import os
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import csv
from multiprocessing import Pool, cpu_count
import threading

# Import pyomo modules
from pyomo.environ import *
from pyomo.opt import SolverFactory

# Thread-safe CSV writer
csv_lock = threading.Lock()


def extract_instance_metadata(data_file):
    """Extract metadata about instance complexity"""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)

        thermal_gens = data['thermal_generators']
        renewable_gens = data['renewable_generators']
        time_periods = data['time_periods']

        # Basic counts
        n_thermal = len(thermal_gens)
        n_renewable = len(renewable_gens)
        n_must_run = sum(1 for g in thermal_gens.values() if g['must_run'] == 1)

        # Startup categories and PWL points
        total_startup_cat = sum(len(g['startup']) for g in thermal_gens.values())
        total_pwl_points = sum(len(g['piecewise_production']) for g in thermal_gens.values())

        # Problem size estimates
        n_binary_vars = n_thermal * time_periods * 3 + total_startup_cat * time_periods
        n_continuous_vars = (n_thermal * 3 + n_renewable) * time_periods + total_pwl_points * time_periods
        n_constraints = time_periods * 2 + n_thermal * time_periods * 10 + total_startup_cat * time_periods + n_thermal * 7

        # Demand and reserves
        peak_demand = max(data['demand'])
        avg_demand = sum(data['demand']) / len(data['demand'])
        total_reserves = sum(data['reserves'])

        return {
            'time_periods': time_periods,
            'n_thermal_gens': n_thermal,
            'n_renewable_gens': n_renewable,
            'n_must_run': n_must_run,
            'total_startup_categories': total_startup_cat,
            'total_pwl_points': total_pwl_points,
            'approx_binary_vars': n_binary_vars,
            'approx_continuous_vars': n_continuous_vars,
            'approx_total_vars': n_binary_vars + n_continuous_vars,
            'approx_constraints': n_constraints,
            'peak_demand': round(peak_demand, 2),
            'avg_demand': round(avg_demand, 2),
            'total_reserves': round(total_reserves, 2),
        }
    except Exception as e:
        print(f"Warning: Could not extract metadata from {data_file}: {e}")
        return {}


def load_and_solve_instance(args):
    """
    Load a UC instance from JSON file and solve it

    Args:
        args: tuple of (data_file, solver_name, ratio_gap, time_limit, threads, verbose, output_file, fieldnames)

    Returns:
        dict: Results including solve time, objective value, and solver status
    """
    data_file, solver_name, ratio_gap, time_limit, threads, verbose, output_file, fieldnames = args

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
        # Extract metadata first
        metadata = extract_instance_metadata(data_file)
        results.update(metadata)

        # Load data
        print(f"\n[{os.path.basename(data_file)}] Loading data...")
        start_load = time.time()
        with open(data_file, 'r') as f:
            data = json.load(f)
        load_time = time.time() - start_load
        print(f"[{os.path.basename(data_file)}] Data loaded in {load_time:.2f}s")

        thermal_gens = data['thermal_generators']
        renewable_gens = data['renewable_generators']

        time_periods = {t+1 : t for t in range(data['time_periods'])}

        gen_startup_categories = {g : list(range(0, len(gen['startup']))) for (g, gen) in thermal_gens.items()}
        gen_pwl_points = {g : list(range(0, len(gen['piecewise_production']))) for (g, gen) in thermal_gens.items()}

        # Build model
        print(f"[{os.path.basename(data_file)}] Building model...")
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
        print(f"[{os.path.basename(data_file)}] Model built in {build_time:.2f}s")

        # Solve
        print(f"[{os.path.basename(data_file)}] Solving with {threads} threads...")

        start_solve = time.time()

        # HiGHS-specific options - use highspy directly
        if solver_name == 'highs':
            try:
                from highspy import Highs, HighsModelStatus

                # Write model to LP file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as tmp:
                    tmp_lp = tmp.name

                m.write(tmp_lp, io_options={'symbolic_solver_labels': True})

                # Create HiGHS solver instance
                h = Highs()
                h.setOptionValue("threads", threads)
                h.setOptionValue("mip_rel_gap", ratio_gap)
                if time_limit:
                    h.setOptionValue("time_limit", float(time_limit))

                # Solve
                h.readModel(tmp_lp)
                h.run()

                # Clean up temp file
                import os as os_module
                try:
                    os_module.unlink(tmp_lp)
                except:
                    pass

                # Get status
                status = h.getModelStatus()

                # Create a fake solve_result object to mimic Pyomo's interface
                class FakeSolverResult:
                    class FakeSolver:
                        def __init__(self, status_val):
                            if status_val == HighsModelStatus.kOptimal:
                                self.status = SolverStatus.ok
                                self.termination_condition = TerminationCondition.optimal
                            elif status_val in [HighsModelStatus.kTimeLimit, HighsModelStatus.kSolutionLimit]:
                                self.status = SolverStatus.ok
                                self.termination_condition = TerminationCondition.feasible
                            else:
                                self.status = SolverStatus.error
                                self.termination_condition = TerminationCondition.unknown

                    class FakeProblem:
                        def __init__(self, obj_val):
                            self.upper_bound = obj_val
                            self.lower_bound = obj_val

                    def __init__(self, status_val, obj_val):
                        self.solver = self.FakeSolver(status_val)
                        self.problem = self.FakeProblem(obj_val)

                # Get objective value directly from HiGHS
                obj_val = None
                if status in [HighsModelStatus.kOptimal, HighsModelStatus.kTimeLimit]:
                    try:
                        obj_val = h.getObjectiveValue()
                    except:
                        pass

                solve_result = FakeSolverResult(status, obj_val)

                # Note: Solution is not loaded back to model variables
                # If needed, value(m.obj) won't work - use solve_result directly

            except ImportError:
                print(f"[{os.path.basename(data_file)}] WARNING: highspy not available, falling back to Pyomo interface")
                solver = SolverFactory(solver_name)
                solve_result = solver.solve(m, tee=verbose)

        # CBC options (fallback)
        elif solver_name == 'cbc':
            solver = SolverFactory(solver_name)
            solver_options = {
                'ratioGap': ratio_gap,
                'threads': threads
            }
            if time_limit:
                solver_options['seconds'] = time_limit
            solve_result = solver.solve(m, options=solver_options, tee=verbose)

        else:
            # Generic solver
            solver = SolverFactory(solver_name)
            solve_result = solver.solve(m, tee=verbose)

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
            # Try to get from solve_result first (works for HiGHS)
            if hasattr(solve_result.problem, 'upper_bound') and solve_result.problem.upper_bound is not None:
                results['objective_value'] = solve_result.problem.upper_bound
            else:
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

        print(f"[{os.path.basename(data_file)}] Status: {results['status']}, Solve time: {solve_time:.2f}s", end="")
        if results['objective_value']:
            print(f", Objective: {results['objective_value']:.2f}")
        else:
            print()

        # Write result to CSV immediately (thread-safe)
        if output_file:
            write_result_to_csv(results, output_file, fieldnames)

    except Exception as e:
        results['error'] = str(e)
        print(f"[{os.path.basename(data_file)}] ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Write error result to CSV
        if output_file:
            write_result_to_csv(results, output_file, fieldnames)

    return results


def write_result_to_csv(result, output_file, fieldnames):
    """Thread-safe CSV writing with proper field ordering"""
    with csv_lock:
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Reorder result to match fieldnames
            ordered_result = {k: result.get(k, '') for k in fieldnames}
            writer.writerow(ordered_result)


def get_solved_instances(output_file):
    """Read CSV and return set of already solved instances"""
    solved = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get('status') in ['optimal', 'feasible']:
                        solved.add(row['instance'])
            print(f"Found {len(solved)} already solved instances in {output_file}")
        except Exception as e:
            print(f"Warning: Could not read existing results from {output_file}: {e}")
    return solved


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
    parser = argparse.ArgumentParser(description='Run UC model on all test instances in parallel')
    parser.add_argument('--solver', type=str, default='cbc',
                       help='Solver to use: highs (default) or cbc')
    parser.add_argument('--gap', type=float, default=0.01,
                       help='MIP gap tolerance (default: 0.01)')
    parser.add_argument('--time-limit', type=int, default=None,
                       help='Time limit per instance in seconds')
    parser.add_argument('--output', type=str, default='new_results_parallel.csv',
                       help='Output CSV file')
    parser.add_argument('--verbose', action='store_true',
                       help='Show solver output')
    parser.add_argument('--instances', nargs='+',
                       help='Specific instances to run (default: all)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of instances to run')
    parser.add_argument('--parallel', type=int, default=8,
                       help='Number of instances to solve in parallel (default: 8)')
    parser.add_argument('--threads-per-instance', type=int, default=2,
                       help='Number of threads per solver instance (default: 2)')

    args = parser.parse_args()

    # Check CPU availability
    total_cpus = cpu_count()
    total_threads_needed = args.parallel * args.threads_per_instance

    print(f"System CPUs: {total_cpus}")
    print(f"Parallel instances: {args.parallel}")
    print(f"Threads per instance: {args.threads_per_instance}")
    print(f"Total threads needed: {total_threads_needed}")

    if total_threads_needed > total_cpus:
        print(f"WARNING: Requested {total_threads_needed} threads but only {total_cpus} CPUs available")
        print(f"         This may cause performance degradation")

    # Find test instances
    if args.instances:
        instances = args.instances
    else:
        instances = find_all_test_instances()

    if args.limit:
        instances = instances[:args.limit]

    print(f"\nFound {len(instances)} test instances")

    # Check for already solved instances (resume functionality)
    solved_instances = get_solved_instances(args.output)
    instances_to_solve = [inst for inst in instances if os.path.basename(inst) not in solved_instances]

    if solved_instances:
        print(f"Skipping {len(solved_instances)} already solved instances")
        print(f"Remaining to solve: {len(instances_to_solve)}")
    else:
        print(f"No existing results found, solving all {len(instances_to_solve)} instances")

    if len(instances_to_solve) == 0:
        print("\nAll instances already solved! Use --output with a different filename to re-solve.")
        return

    print(f"\nSolver: {args.solver}")
    print(f"MIP gap: {args.gap}")
    if args.time_limit:
        print(f"Time limit: {args.time_limit}s per instance")
    print()

    # Initialize CSV file with header (including metadata fields)
    fieldnames = ['instance', 'status', 'solve_time', 'build_time', 'load_time',
                 'total_time', 'objective_value', 'gap', 'error', 'file_path',
                 'time_periods', 'n_thermal_gens', 'n_renewable_gens', 'n_must_run',
                 'total_startup_categories', 'total_pwl_points',
                 'approx_binary_vars', 'approx_continuous_vars', 'approx_total_vars',
                 'approx_constraints', 'peak_demand', 'avg_demand', 'total_reserves']

    # Create file with header if it doesn't exist
    if not os.path.exists(args.output):
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Prepare arguments for parallel execution (only for unsolved instances)
    solve_args = [
        (instance, args.solver, args.gap, args.time_limit,
         args.threads_per_instance, args.verbose, args.output, fieldnames)
        for instance in instances_to_solve
    ]

    # Run in parallel
    start_time = time.time()

    print(f"Starting parallel execution with {args.parallel} workers...\n")

    with Pool(processes=args.parallel) as pool:
        all_results = pool.map(load_and_solve_instance, solve_args)

    total_time = time.time() - start_time

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\nResults saved to {args.output}")
    print(f"\nTotal instances processed: {len(instances_to_solve)}")
    print(f"Total instances (including skipped): {len(instances)}")
    print(f"Total wall-clock time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

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
            print(f"  Total CPU time: {sum(solve_times):.2f}s")
            print(f"  Speedup: {sum(solve_times)/total_time:.2f}x")

    if failed:
        print(f"\nFailed instances:")
        for r in failed:
            print(f"  {r['instance']}: {r['status']} - {r['error'] if r['error'] else 'N/A'}")


if __name__ == "__main__":
    main()