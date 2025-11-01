"""
Relax-and-Fix solver for Unit Commitment problem
"""
import time

from pyomo.environ import Binary, UnitInterval, Var, value
from pyomo.opt import SolverFactory, TerminationCondition


def _get_time_index_position(var):
    """
    Определить позицию индекса времени для переменной UC модели

    Переменные в UC модели:
    - ug, vg, wg: (generator, time) -> time at index 1
    - dg: (generator, startup_category, time) -> time at index 2

    Returns:
        int: Position of time index in variable indices
    """
    # Проверяем количество индексов через первый элемент
    first_idx = next(iter(var))
    if len(first_idx) == 2:
        return 1  # (g, t)
    elif len(first_idx) == 3:
        return 2  # (g, s, t)
    else:
        raise ValueError(f"Unexpected index structure for variable")


def _find_binary_variables(model):
    """
    Автоматически найти все бинарные переменные в модели

    Returns:
        list: List of (var_object, time_idx_pos) tuples
    """
    binary_vars = []
    for component in model.component_objects(ctype=Var):
        try:
            # Проверяем domain первой переменной
            first_var = next(iter(component.values()))
            if first_var.domain == Binary:
                time_idx_pos = _get_time_index_position(component)
                binary_vars.append((component, time_idx_pos))
        except (StopIteration, ValueError):
            # Переменная пустая, без индексов или неподходящая структура
            pass

    return binary_vars


def _verify_solution_feasibility(old_model, data, model_builder, solver_name="appsi_highs", gap=0.01, verbose=False):
    """
    Проверить допустимость найденного решения в исходной (немодифицированной) задаче

    Создаёт НОВУЮ модель из исходных данных (без модификаций),
    копирует в неё значения переменных из решения и проверяет допустимость.

    Args:
        old_model: Pyomo ConcreteModel after relax-and-fix solution (with rounded variables)
        data: Original problem data (JSON dict)
        model_builder: Function to build fresh model from data (e.g., build_uc_model)
        solver_name: Solver to use for verification
        gap: MIP gap tolerance
        verbose: Show verification details

    Returns:
        dict: {
            'feasible': bool,
            'objective': float or None,
            'original_objective': float (objective from rounded solution),
            'gap': float (absolute difference in objectives)
        }
    """
    if verbose:
        print("\n  Verifying solution feasibility in FRESH model...")

    original_objective = value(old_model.obj)

    # Сохранить все значения переменных из старой модели
    if verbose:
        print("    Extracting variable values from relax-and-fix solution...")

    var_values = {}
    for component in old_model.component_objects(ctype=Var):
        var_values[component.name] = {}
        for idx in component:
            var_values[component.name][idx] = value(component[idx])

    # Создать НОВУЮ модель из исходных данных
    if verbose:
        print("    Building fresh model from original data...")

    new_model = model_builder(data)

    # Установить значения переменных в новую модель
    if verbose:
        print("    Setting variable values in fresh model...")

    vars_set = 0
    for var_name, values in var_values.items():
        if hasattr(new_model, var_name):
            new_var = getattr(new_model, var_name)
            for idx, val in values.items():
                if idx in new_var:
                    new_var[idx].value = val
                    vars_set += 1

    if verbose:
        print(f"    Set {vars_set} variable values")

    # Зафиксировать все бинарные переменные на установленных значениях
    # (это делает задачу LP - быстрая проверка допустимости)
    new_binary_vars = _find_binary_variables(new_model)
    for var, time_idx_pos in new_binary_vars:
        for idx in var:
            var[idx].fix()

    # Решить задачу с зафиксированными переменными (проверка допустимости)
    if verbose:
        print("    Solving fresh model with fixed binary variables...")

    solver = SolverFactory(solver_name)

    if hasattr(solver, 'config'):
        # APPSI solvers (appsi_highs, etc)
        solver.config.mip_gap = gap
        result = solver.solve(new_model, tee=False)
    else:
        # Legacy solvers (cbc, etc)
        result = solver.solve(new_model, options={'ratioGap': gap}, tee=False)

    # Проверить результаты
    is_feasible = result.solver.termination_condition == TerminationCondition.optimal
    verified_objective = value(new_model.obj) if is_feasible else None
    objective_gap = abs(verified_objective - original_objective) if verified_objective is not None else float('inf')

    if verbose:
        print()
        if is_feasible:
            print(f"  Solution is FEASIBLE in fresh model")
            print(f"    Relax-and-fix objective: {original_objective:.2f}")
            print(f"    Fresh model objective:   {verified_objective:.2f}")
            print(f"    Absolute gap: {objective_gap:.2f} ({abs(objective_gap / original_objective) * 100:.4f}%)")
        else:
            print(f"  Solution is INFEASIBLE in fresh model")
            print(f"    Termination condition: {result.solver.termination_condition}")
            print(f"    This indicates relax-and-fix found an invalid solution!")

    return {
        'feasible': is_feasible,
        'objective': verified_objective,
        'original_objective': original_objective,
        'gap': objective_gap
    }


# todo реализовать внутренний цикл по генераторам (перебирать по несколько штук, по мощности)
# todo почему решение в первом периоде занимает много, а потом меньше?
def solve_relax_and_fix(model, window_size, window_step, gap, solver_name,
                        verbose=False, verify_solution=True, data=None, model_builder=None):
    """
    Solve UC model using Relax-and-Fix approach

    Основная идея:
    1. Разбить временной горизонт на окна (windows)
    2. Для текущего окна переменные остаются бинарными
    3. Для будущих периодов переменные релаксируются (становятся непрерывными в [0,1])
    4. Решаем подзадачу
    5. Фиксируем решение для части окна
    6. Двигаем окно дальше

    Args:
        model: Pyomo ConcreteModel with UC formulation
        window_size: Size of time window for binary variables
        window_step: Step size for moving window
        solver_name: Solver name for factory
        gap: MIP gap tolerance
        verbose: Show solver output
        verify_solution: Verify final solution feasibility in original problem (default: True)
        data: Original problem data (required if verify_solution=True)
        model_builder: Function to build model from data (required if verify_solution=True)

    Returns:
        dict: Results with solve_time, objective, status, and optional verification info
    """
    start_time = time.time()

    # Get time periods (используем как reference)
    time_periods = sorted(list(list(model.ug.index_set().subsets())[1]))
    num_periods = len(time_periods)

    # Автоматически находим все бинарные переменные
    binary_vars = _find_binary_variables(model)

    if verbose:
        print(f"  Found {len(binary_vars)} binary variable types to relax")
        for var, time_pos in binary_vars:
            print(f"    - {var.name}: time index at position {time_pos}")

    # Relax-and-Fix iterations
    for start in range(0, num_periods, window_step):
        end = min(start + window_size, num_periods)
        step = min(start + window_step, num_periods)

        if end == num_periods:
            step = end

        if verbose:
            print(f"  Window [{start}:{end}], fixing [{start}:{step}]")

        window_periods = set(time_periods[start:end])
        fix_periods = set(time_periods[start:step])

        # Обрабатываем все бинарные переменные единообразно
        for var, time_idx_pos in binary_vars:
            for idx in var:
                time_period = idx[time_idx_pos]

                # Переменные в окне - бинарные, остальные - непрерывные
                if time_period in window_periods:
                    var[idx].domain = Binary
                elif not var[idx].is_fixed():
                    var[idx].domain = UnitInterval

        # Solve (create new solver each iteration)
        iter_start = time.time()
        solver = SolverFactory(solver_name)

        # Set gap depending on solver type and solve
        if hasattr(solver, 'config'):
            # APPSI solvers (appsi_highs, etc)
            solver.config.mip_gap = gap
            result = solver.solve(model)
        else:
            # Legacy solvers (cbc, etc)
            result = solver.solve(model, options={'ratioGap': gap})

        # Check iteration result
        iter_feasible = result.solver.termination_condition == TerminationCondition.optimal

        if verbose:
            status_str = "optimal" if iter_feasible else str(result.solver.termination_condition)
            print(f"  Iteration solved in {time.time() - iter_start:.2f}s, obj={value(model.obj):.2f}, status={status_str}")

        if not iter_feasible and verbose:
            print(f"  WARNING: Iteration did not find optimal solution (continuing anyway)")

        # Фиксируем переменные, которые покидают окно
        for var, time_idx_pos in binary_vars:
            for idx in var:
                if idx[time_idx_pos] in fix_periods and not var[idx].is_fixed():
                    var[idx].fix()

        if step == num_periods:
            break

    solve_time = time.time() - start_time

    # Prepare result
    result = {
        'solve_time': solve_time,
        'objective': value(model.obj),
        'status': 'completed',
    }

    if verify_solution:
        if data is None or model_builder is None:
            raise ValueError(
                "verify_solution=True requires 'data' and 'model_builder' arguments. "
                "Pass the original problem data and model building function."
            )

        verification = _verify_solution_feasibility(
            old_model=model,
            data=data,
            model_builder=model_builder,
            solver_name=solver_name,
            gap=gap,
            verbose=verbose
        )
        result['verification'] = verification
        result['verified_objective'] = verification['objective']
        result['objective_gap'] = verification['gap']
        result['feasible'] = verification['feasible']

    return result
