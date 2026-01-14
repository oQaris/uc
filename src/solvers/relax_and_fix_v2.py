"""
Optimized Relax-and-Fix solver with persistent solver and backward unfixing
"""
import time
from pyomo.environ import value
from pyomo.opt import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from .rf_utils import (
    find_binary_variables, get_generators_sorted_by_power,
    set_variable_domains, fix_variables, unfix_variables_in_window,
    fix_future_variables_to_zero, refix_periods
)
from .rf_backward_unfix import backward_unfix_for_startup
from .rf_lookahead import calculate_generator_lookahead, manage_constraints_for_window
from .diagnostics import diagnose_infeasibility


class RelaxAndFixSolver:
    """Persistent solver for Relax-and-Fix with adaptive lookahead"""

    def __init__(self, model, data, gap=0.01, verbose=False, generator_sort_function=None):
        self.model = model
        self.data = data
        self.gap = gap
        self.verbose = verbose
        self.generator_sort_function = generator_sort_function

        # Initialize persistent solver
        self.solver = Highs()
        self.solver.highs_options.update({
            "presolve": "on",
            "threads": 1,
            "output_flag": False,
            "log_to_console": False,
            "mip_rel_gap": gap,
        })
        self.solver.set_instance(model)

        # Extract model info
        time_periods_list = sorted(list(list(model.ug.index_set().subsets())[1]))
        self.time_periods = time_periods_list
        self.num_periods = len(time_periods_list)
        self.binary_vars = find_binary_variables(model)

    def solve(self, window_size, window_step, generators_per_iteration=None,
              use_limited_horizon=True, verify_solution=False, model_builder=None):
        """
        Execute Relax-and-Fix algorithm

        Args:
            window_size: Size of time window for binary variables
            window_step: Step size for moving window
            generators_per_iteration: Generators per batch (None = all)
            use_limited_horizon: Use adaptive lookahead
            verify_solution: Verify final solution
            model_builder: Function to rebuild model for verification
        """
        start_time = time.time()

        # Get sorted generators (custom function or default by power)
        if self.generator_sort_function is not None:
            generators_sorted = self.generator_sort_function(self.data)
        else:
            generators_sorted = get_generators_sorted_by_power(self.data)
        num_generators = len(generators_sorted)

        if generators_per_iteration is None:
            generators_per_iteration = num_generators

        if self.verbose:
            print(f"  Generator decomposition: {num_generators} total, {generators_per_iteration} per batch")
            print(f"  Lookahead mode: {'adaptive' if use_limited_horizon else 'full horizon'}")

        # Main Relax-and-Fix loop
        for start in range(0, self.num_periods, window_step):
            end = min(start + window_size, self.num_periods)
            step = min(start + window_step, self.num_periods)
            if end == self.num_periods:
                step = end

            window_periods = set(self.time_periods[start:end])
            fix_periods = set(self.time_periods[start:step])

            # Backward unfixing
            unfixed_periods = set()
            if start > 0:
                unfixed_periods = backward_unfix_for_startup(
                    self.model, self.time_periods[start], self.data, self.verbose
                )

            # Manage lookahead and constraints
            if use_limited_horizon:
                generator_lookahead = calculate_generator_lookahead(
                    self.model, start, step, self.data, self.num_periods
                )
                max_lookahead = max(generator_lookahead.values())
                lookahead_periods = set(self.time_periods[step:min(max_lookahead, self.num_periods)])
                future_periods = set(self.time_periods[max_lookahead:]) if max_lookahead < self.num_periods else set()

                if self.verbose:
                    print(f"  Window [{start}:{end}], fix [{start}:{step}], lookahead [{step}:{max_lookahead}]")

                if lookahead_periods:
                    unfix_variables_in_window(self.model, lookahead_periods, self.binary_vars, self.verbose)
                if future_periods:
                    fix_future_variables_to_zero(self.model, future_periods, self.binary_vars,
                                                 generator_lookahead, self.verbose)

                manage_constraints_for_window(self.model, generator_lookahead, self.num_periods, self.verbose)
            else:
                if self.verbose:
                    print(f"  Window [{start}:{end}], fix [{start}:{step}]")

            # Generator batches
            for gen_start_idx in range(0, num_generators, generators_per_iteration):
                gen_end_idx = min(gen_start_idx + generators_per_iteration, num_generators)
                current_gen_batch = set(generators_sorted[gen_start_idx:gen_end_idx])

                if self.verbose and generators_per_iteration < num_generators:
                    print(f"    Generator batch [{gen_start_idx}:{gen_end_idx}]")

                # Set domains and solve
                set_variable_domains(self.binary_vars, current_gen_batch, window_periods)
                result, solve_time, is_optimal = self._solve_subproblem()

                # Check result
                if result.solver.termination_condition in [TerminationCondition.infeasible,
                                                          TerminationCondition.infeasibleOrUnbounded]:
                    iteration_info = {
                        'start': start, 'end': end, 'step': step,
                        'gen_start': gen_start_idx, 'gen_end': gen_end_idx
                    }
                    diagnose_infeasibility(self.model, self.data, iteration_info, export_model=True)
                    raise RuntimeError(f"Infeasible subproblem at window [{start}:{end}], batch [{gen_start_idx}:{gen_end_idx}]")

                if self.verbose:
                    indent = "    " if generators_per_iteration < num_generators else "  "
                    obj_str = f"obj={value(self.model.obj):.2f}" if is_optimal else "obj=N/A"
                    print(f"{indent}Solved in {solve_time:.2f}s, {obj_str}, status={'optimal' if is_optimal else 'feasible'}")

                # Fix variables
                fix_variables(self.binary_vars, current_gen_batch, fix_periods)

            # Re-fix backward-unfixed periods
            if unfixed_periods:
                refix_periods(self.model, unfixed_periods, self.verbose)

            if step == self.num_periods:
                break

        total_time = time.time() - start_time
        result = {
            'solve_time': total_time,
            'objective': value(self.model.obj),
            'status': 'completed',
        }

        if verify_solution and model_builder:
            from .relax_and_fix import _verify_solution_feasibility
            verification = _verify_solution_feasibility(
                self.model, self.data, model_builder, "appsi_highs", self.gap, self.verbose
            )
            result['verification'] = verification

        return result

    def _solve_subproblem(self):
        """Solve current subproblem using persistent solver"""
        iter_start = time.time()

        try:
            self.solver.update()
            res = self.solver.solve(self.model)
            solve_time = time.time() - iter_start
            is_optimal = res.termination_condition == TerminationCondition.optimal
            return res, solve_time, is_optimal
        except RuntimeError as e:
            if "feasible solution was not found" in str(e):
                from pyomo.opt import SolverResults
                result = SolverResults()
                result.solver.termination_condition = TerminationCondition.infeasible
                return result, time.time() - iter_start, False
            raise


def solve_relax_and_fix_v2(model, data, window_size, window_step, gap=0.01,
                            verbose=False, generators_per_iteration=None,
                            use_limited_horizon=True, verify_solution=False,
                            model_builder=None, generator_sort_function=None):
    """
    Optimized Relax-and-Fix with persistent solver

    Args:
        model: Pyomo ConcreteModel
        data: Original problem data
        window_size: Time window size
        window_step: Window step size
        gap: MIP gap tolerance
        verbose: Print progress
        generators_per_iteration: Generators per batch (None = all)
        use_limited_horizon: Use adaptive lookahead
        verify_solution: Verify final solution
        model_builder: Model builder function for verification
        generator_sort_function: Custom function(data) -> list to sort generators
                                (None = sort by max power descending)

    Returns:
        dict: Results with solve_time, objective, status
    """
    solver = RelaxAndFixSolver(model, data, gap, verbose, generator_sort_function)
    return solver.solve(window_size, window_step, generators_per_iteration,
                       use_limited_horizon, verify_solution, model_builder)
