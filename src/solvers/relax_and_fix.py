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


def _get_generators_sorted_by_power(data):
    """
    Получить список генераторов, отсортированных по максимальной мощности (убывание)

    Args:
        data: dict - исходные данные задачи с ключом 'thermal_generators'

    Returns:
        list: Список имен генераторов, отсортированных по power_output_maximum (от большей к меньшей)
    """
    if data is None:
        raise ValueError("data is required for generator sorting")

    thermal_gens = data.get("thermal_generators", {})

    # Создаем список (имя_генератора, макс_мощность)
    gen_power_pairs = [(g, gen_data.get("power_output_maximum", 0.0))
                       for g, gen_data in thermal_gens.items()]

    # Сортируем по мощности (убывание)
    gen_power_pairs.sort(key=lambda x: x[1], reverse=True)

    # Возвращаем только имена
    return [g for g, _ in gen_power_pairs]


def _set_variable_domains(binary_vars, binary_generators, binary_periods):
    """
    Установить домены переменных для текущей подзадачи

    Args:
        binary_vars: список (var, time_idx_pos) - бинарные переменные модели
        binary_generators: set - генераторы, которые должны быть бинарными
        binary_periods: set - временные периоды, которые должны быть бинарными
    """
    for var, time_idx_pos in binary_vars:
        for idx in var:
            time_period = idx[time_idx_pos]
            generator = idx[0]  # Для всех переменных UC модели генератор на позиции 0

            # Бинарные: генератор в партии И период в окне
            if generator in binary_generators and time_period in binary_periods:
                var[idx].domain = Binary
            elif not var[idx].is_fixed():
                var[idx].domain = UnitInterval


def _solve_subproblem(model, solver_name, gap):
    """
    Решить текущую подзадачу

    Args:
        model: Pyomo модель
        solver_name: имя решателя
        gap: MIP gap tolerance

    Returns:
        tuple: (result, solve_time, is_optimal)
    """
    iter_start = time.time()
    solver = SolverFactory(solver_name)

    # Установка gap в зависимости от типа решателя
    if hasattr(solver, 'config'):
        # APPSI solvers (appsi_highs, etc)
        solver.config.mip_gap = gap
        result = solver.solve(model)
    else:
        # Legacy solvers (cbc, etc)
        result = solver.solve(model, options={'ratioGap': gap})

    solve_time = time.time() - iter_start
    is_optimal = result.solver.termination_condition == TerminationCondition.optimal

    return result, solve_time, is_optimal


def _fix_variables(binary_vars, fix_generators, fix_periods):
    """
    Зафиксировать переменные для текущей подзадачи

    Args:
        binary_vars: список (var, time_idx_pos) - бинарные переменные модели
        fix_generators: set - генераторы, которые нужно зафиксировать
        fix_periods: set - периоды, которые нужно зафиксировать
    """
    for var, time_idx_pos in binary_vars:
        for idx in var:
            generator = idx[0]  # Генератор на первой позиции
            time_period = idx[time_idx_pos]

            # Фиксируем: генератор в партии И период в окне фиксации И не зафиксирован
            if (generator in fix_generators and
                    time_period in fix_periods and
                    not var[idx].is_fixed()):
                var[idx].fix()


def _verify_solution_feasibility(old_model, data, model_builder, solver_name, gap, verbose):
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


def solve_relax_and_fix(model, window_size, window_step, gap, solver_name,
                        verbose=False, verify_solution=True, data=None, model_builder=None,
                        generators_per_iteration=None, generator_sort_function=None):
    """
    Solve UC model using Relax-and-Fix approach with optional generator decomposition

    Алгоритм:
    1. Разбить временной горизонт на окна (window_size, window_step)
    2. Разбить генераторы на партии (generators_per_iteration)
       - Если generators_per_iteration=None: все генераторы сразу (классический R&F)
       - Генераторы сортируются по максимальной мощности (убывание)
    3. Для каждого временного окна:
       - Для каждой партии генераторов:
         - Переменные текущей партии в текущем окне - бинарные
         - Остальные переменные - релаксированы [0,1]
         - Решить подзадачу
         - Зафиксировать переменные партии в окне фиксации
    4. Двигаем окно дальше

    Args:
        model: Pyomo ConcreteModel with UC formulation
        window_size: Size of time window for binary variables
        window_step: Step size for moving window
        gap: MIP gap tolerance
        solver_name: Solver name for factory
        verbose: Show solver output
        verify_solution: Verify final solution feasibility in original problem (default: True)
        data: Original problem data (required for generator sorting and verification)
        model_builder: Function to build model from data (required if verify_solution=True)
        generators_per_iteration: Number of generators per iteration (None = all at once)
        generator_sort_function: Custom function(data) -> list to sort generators (None = by power desc)

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

    # Получить отсортированный список всех генераторов
    if data is None:
        raise ValueError("'data' argument is required for generator sorting")

    # Используем кастомную функцию сортировки или дефолтную (по мощности)
    if generator_sort_function is not None:
        generators_sorted = generator_sort_function(data)
    else:
        generators_sorted = _get_generators_sorted_by_power(data)

    num_generators = len(generators_sorted)

    # Если generators_per_iteration не задан, берем все генераторы сразу (классический R&F)
    if generators_per_iteration is None:
        generators_per_iteration = num_generators
        if verbose:
            print(f"  Classic Relax-and-Fix: {num_generators} generators at once")
    else:
        if verbose:
            print(f"  Generator-wise decomposition: {num_generators} generators, "
                  f"{generators_per_iteration} per iteration")
            print(f"    Generators sorted by power (top 5): {generators_sorted[:5]}")

    # Основной цикл Relax-and-Fix (единая логика для всех стратегий)
    for start in range(0, num_periods, window_step):
        end = min(start + window_size, num_periods)
        step = min(start + window_step, num_periods)

        # Последнее окно - фиксируем все до конца
        if end == num_periods:
            step = end

        window_periods = set(time_periods[start:end])
        fix_periods = set(time_periods[start:step])

        if verbose:
            print(f"  Time Window [{start}:{end}], fixing [{start}:{step}]")

        # Внутренний цикл по партиям генераторов
        for gen_start_idx in range(0, num_generators, generators_per_iteration):
            gen_end_idx = min(gen_start_idx + generators_per_iteration, num_generators)
            current_gen_batch = set(generators_sorted[gen_start_idx:gen_end_idx])

            # Вывод информации о партии (если декомпозиция включена)
            if verbose and generators_per_iteration < num_generators:
                print(f"    Generator batch [{gen_start_idx}:{gen_end_idx}] ({len(current_gen_batch)} gens)")

            # 1. Установить домены переменных
            _set_variable_domains(binary_vars, current_gen_batch, window_periods)

            # 2. Решить подзадачу
            result, solve_time, is_optimal = _solve_subproblem(model, solver_name, gap)

            # 3. Вывод результата
            if verbose:
                status_str = "optimal" if is_optimal else str(result.solver.termination_condition)
                indent = "    " if generators_per_iteration < num_generators else "  "
                print(f"{indent}Solved in {solve_time:.2f}s, obj={value(model.obj):.2f}, status={status_str}")

                if not is_optimal:
                    print(f"{indent}WARNING: Iteration did not find optimal solution (continuing anyway)")

            # 4. Зафиксировать переменные
            _fix_variables(binary_vars, current_gen_batch, fix_periods)

        # Выход после последнего окна
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
