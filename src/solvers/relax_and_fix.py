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


def _calculate_generator_specific_lookahead(model, boundary_period, fix_end_period, data, num_periods):
    """
    Определить индивидуальный lookahead для каждого генератора на основе состояния на границе

    Анализирует последний зафиксированный период (boundary_period - 1) и определяет,
    до какого периода нужно видеть ограничения для поддержки консистентности с min_uptime/min_downtime.

    Args:
        model: Pyomo ConcreteModel - модель с зафиксированными переменными до boundary_period
        boundary_period: int - первый период текущего окна (граница между зафиксированным и новым)
        fix_end_period: int - конец окна фиксации текущей итерации (step)
        data: dict - исходные данные задачи с ключом 'thermal_generators'
        num_periods: int - общее количество периодов

    Returns:
        dict: {generator_name: required_end_period} для ВСЕХ генераторов
              Минимальный lookahead = fix_end_period + 1 (для ramping constraints)
    """
    thermal_gens = data.get("thermal_generators", {})
    generator_lookahead = {}

    # Минимальный lookahead для всех = +1 период ПОСЛЕ фиксации (для ramping constraints)
    min_lookahead_end = min(fix_end_period + 1, num_periods)

    # Определить оставшийся горизонт
    remaining_horizon = num_periods - fix_end_period

    # Вычислить максимальное ограничение среди всех генераторов
    # Это нужно для определения, когда использовать полный горизонт
    max_constraint_global = 0
    for gen_data in thermal_gens.values():
        min_uptime = gen_data.get("time_up_minimum", 0)
        min_downtime = gen_data.get("time_down_minimum", 0)
        max_constraint_global = max(max_constraint_global, min_uptime, min_downtime)

    # Если оставшийся горизонт меньше или равен максимальному ограничению,
    # используем полный горизонт для всех генераторов
    # Это гарантирует, что все генераторы видят достаточно вперед для своих ограничений
    use_full_horizon = remaining_horizon <= max_constraint_global

    for g, gen_data in thermal_gens.items():
        min_uptime = gen_data.get("time_up_minimum", 0)
        min_downtime = gen_data.get("time_down_minimum", 0)

        if use_full_horizon:
            # Близко к концу горизонта - используем полный lookahead для всех
            generator_lookahead[g] = num_periods
        else:
            # Консервативный lookahead: генератор может включиться/выключиться в последний период окна фиксации
            # В этом случае нужен lookahead до fix_end_period + max(min_uptime, min_downtime)
            max_constraint = max(min_uptime, min_downtime)
            constraint_based = min(fix_end_period + max_constraint,
                                   num_periods) if max_constraint > 0 else min_lookahead_end

            # Минимальный lookahead: хотя бы половина оставшегося горизонта
            # Это гарантирует, что генераторы с малыми ограничениями все равно видят достаточно вперед
            min_reasonable_lookahead = min(fix_end_period + remaining_horizon // 2, num_periods)

            generator_lookahead[g] = max(constraint_based, min_reasonable_lookahead)

        # Получить состояние на границе (последний зафиксированный период)
        if boundary_period == 0:
            # Самое первое окно - используем начальное состояние
            initial_status = gen_data.get("initial_status", 0)
            ug_at_boundary = 1 if initial_status > 0 else 0
        else:
            try:
                ug_at_boundary = round(value(model.ug[g, boundary_period - 1]))
            except:
                ug_at_boundary = 0

        # Случай 1: Генератор работает на границе - проверить min_uptime
        if ug_at_boundary == 1 and min_uptime > 0:
            # Посчитать сколько периодов подряд он уже работает к моменту boundary
            uptime_so_far = 1
            for t in range(boundary_period - 2, -1, -1):
                try:
                    ug_value = round(value(model.ug[g, t]))
                    if ug_value == 1:
                        uptime_so_far += 1
                    else:
                        break
                except:
                    break

            # Если не завершил min_uptime, нужен extended lookahead
            if uptime_so_far < min_uptime:
                remaining = min_uptime - uptime_so_far
                required_end = min(fix_end_period + remaining, num_periods)
                generator_lookahead[g] = max(generator_lookahead[g], required_end)

        # Случай 2: Генератор выключен на границе - проверить min_downtime
        elif ug_at_boundary == 0 and min_downtime > 0:
            # Посчитать сколько периодов подряд он уже не работает
            downtime_so_far = 1
            for t in range(boundary_period - 2, -1, -1):
                try:
                    ug_value = round(value(model.ug[g, t]))
                    if ug_value == 0:
                        downtime_so_far += 1
                    else:
                        break
                except:
                    break

            # Если не завершил min_downtime, нужен extended lookahead
            if downtime_so_far < min_downtime:
                remaining = min_downtime - downtime_so_far
                required_end = min(fix_end_period + remaining, num_periods)
                generator_lookahead[g] = max(generator_lookahead[g], required_end)

    return generator_lookahead


def _manage_constraints_for_window(model, generator_lookahead, num_periods, verbose=False):
    """
    Активировать/деактивировать ограничения с учетом индивидуального lookahead КАЖДОГО генератора

    Системные ограничения (demand, reserves) активны до max(generator_lookahead).
    Генераторные ограничения активны индивидуально до generator_lookahead[g].

    Args:
        model: Pyomo ConcreteModel
        generator_lookahead: dict - {generator_name: required_end_period} для ВСЕХ генераторов
        num_periods: int - общее количество периодов
        verbose: bool - выводить информацию о деактивации
    """
    # Определить максимальный период активности среди всех генераторов
    max_lookahead = max(generator_lookahead.values()) if generator_lookahead else num_periods

    # СИСТЕМНЫЕ ограничения (t) - активны до max_lookahead
    system_constraints = [
        (model.demand, 0),
        (model.reserves, 0),
    ]

    # ГЕНЕРАТОРНЫЕ ограничения
    # Формат: (constraint_object, time_index_position, generator_index_position)
    generator_constraints = [
        # (g, t) - генератор на позиции 0, время на позиции 1
        (model.mustrun, 1, 0),
        (model.logical, 1, 0),
        (model.uptime, 1, 0),
        (model.downtime, 1, 0),
        (model.startup_select, 1, 0),
        (model.gen_limit1, 1, 0),
        (model.gen_limit2, 1, 0),
        (model.ramp_up, 1, 0),
        (model.ramp_down, 1, 0),
        (model.power_select, 1, 0),  # PWL
        (model.cost_select, 1, 0),  # PWL
        (model.on_select, 1, 0),  # PWL
        # (g, s, t) - генератор на позиции 0, время на позиции 2
        (model.startup_allowed, 2, 0),
    ]

    deactivated_count = 0
    activated_count = 0

    # 1. Управление системными ограничениями (до max_lookahead)
    for constraint, time_idx_pos in system_constraints:
        for idx in constraint:
            time_period = idx

            if time_period <= max_lookahead:
                if not constraint[idx].active:
                    constraint[idx].activate()
                    activated_count += 1
            else:
                if constraint[idx].active:
                    constraint[idx].deactivate()
                    deactivated_count += 1

    # 2. Управление генераторными ограничениями (индивидуально по генераторам)
    for constraint, time_idx_pos, gen_idx_pos in generator_constraints:
        for idx in constraint:
            if not isinstance(idx, tuple):
                continue

            generator = idx[gen_idx_pos]
            time_period = idx[time_idx_pos]

            # Каждый генератор имеет свой индивидуальный lookahead
            required_end = generator_lookahead.get(generator, max_lookahead)
            should_activate = time_period <= required_end

            if should_activate:
                if not constraint[idx].active:
                    constraint[idx].activate()
                    activated_count += 1
            else:
                if constraint[idx].active:
                    constraint[idx].deactivate()
                    deactivated_count += 1

    if verbose and (deactivated_count > 0 or activated_count > 0):
        print(f"    Constraints: activated {activated_count}, deactivated {deactivated_count}")
        print(f"    Max generator lookahead: period {max_lookahead}")
        # Показать генераторы с extended lookahead
        extended_gens = [(g, la) for g, la in generator_lookahead.items() if la > min(generator_lookahead.values())]
        if extended_gens and len(extended_gens) <= 10:
            print(
                f"    Extended lookahead gens: {', '.join([f'{g}->{la}' for g, la in sorted(extended_gens, key=lambda x: x[1], reverse=True)])}")


def _fix_future_variables_to_zero(model, future_periods, binary_vars, generator_lookahead, verbose=False):
    """
    Селективно зафиксировать переменные для дальних будущих периодов на 0

    НЕ фиксирует переменные критических генераторов в их extended lookahead периодах.
    Это позволяет избежать конфликтов с min_uptime/min_downtime ограничениями.

    Args:
        model: Pyomo ConcreteModel
        future_periods: set - периоды вне базового горизонта релаксации
        binary_vars: list - список (var, time_idx_pos) бинарных переменных
        generator_lookahead: dict - {generator_name: required_end_period} для критических генераторов
        verbose: bool - выводить информацию

    Note:
        Также фиксирует непрерывные переменные (pg, rg, cg) на 0 для этих периодов.
    """
    fixed_count = 0
    skipped_count = 0

    # Фиксируем бинарные переменные (ug, vg, wg, dg)
    for var, time_idx_pos in binary_vars:
        for idx in var:
            generator = idx[0]  # Генератор всегда на позиции 0
            time_period = idx[time_idx_pos]

            if time_period not in future_periods or var[idx].is_fixed():
                continue

            # Проверить: не попадает ли период в extended lookahead критического генератора
            if generator in generator_lookahead:
                if time_period <= generator_lookahead[generator]:
                    # Критический генератор в extended lookahead - НЕ фиксируем
                    skipped_count += 1
                    continue

            # Фиксируем на 0
            var[idx].fix(0)
            fixed_count += 1

    # Также фиксируем непрерывные переменные генераторов на 0
    continuous_vars_to_fix = [
        (model.pg, 1),  # Power above minimum (g, t)
        (model.rg, 1),  # Reserves (g, t)
        (model.cg, 1),  # Cost (g, t)
        (model.lg, 2),  # PWL weights (g, l, t) - время на позиции 2
    ]

    for var, time_idx_pos in continuous_vars_to_fix:
        for idx in var:
            generator = idx[0]
            time_period = idx[time_idx_pos]

            if time_period not in future_periods or var[idx].is_fixed():
                continue

            # Проверить extended lookahead
            if generator in generator_lookahead:
                if time_period <= generator_lookahead[generator]:
                    skipped_count += 1
                    continue

            var[idx].fix(0)
            fixed_count += 1

    if verbose and (fixed_count > 0 or skipped_count > 0):
        print(f"    Fixed {fixed_count} future variables to 0 (skipped {skipped_count} critical gen variables)")


def _unfix_variables_in_window(model, window_periods, binary_vars, verbose=False):
    """
    Освободить переменные в текущем окне (отменить фиксацию)

    Используется при движении окна вперед для активации новых периодов.

    Args:
        model: Pyomo ConcreteModel
        window_periods: set - периоды, которые нужно освободить
        binary_vars: list - список (var, time_idx_pos) бинарных переменных
        verbose: bool - выводить информацию
    """
    unfixed_count = 0

    # Освобождаем бинарные переменные
    for var, time_idx_pos in binary_vars:
        for idx in var:
            time_period = idx[time_idx_pos]
            if time_period in window_periods and var[idx].is_fixed():
                var[idx].unfix()
                unfixed_count += 1

    # Освобождаем непрерывные переменные
    continuous_vars = [
        (model.pg, 1),
        (model.rg, 1),
        (model.cg, 1),
        (model.lg, 2),
    ]

    for var, time_idx_pos in continuous_vars:
        for idx in var:
            time_period = idx[time_idx_pos]
            if time_period in window_periods and var[idx].is_fixed():
                var[idx].unfix()
                unfixed_count += 1

    if verbose and unfixed_count > 0:
        print(f"    Unfixed {unfixed_count} variables in window")


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
        # Пытаемся установить load_solution=False если параметр доступен
        try:
            solver.config.load_solution = False
        except (ValueError, AttributeError):
            # Параметр может отсутствовать в некоторых версиях
            pass

        try:
            result = solver.solve(model)
        except RuntimeError as e:
            # HiGHS может бросить RuntimeError если решение не найдено
            # В этом случае все равно возвращаем результат с termination_condition
            if "feasible solution was not found" in str(e):
                # Получаем results напрямую из solver
                result = solver.results
                solve_time = time.time() - iter_start
                return result, solve_time, False
            else:
                raise
    else:
        # Legacy solvers (cbc, etc)
        result = solver.solve(model, options={'ratioGap': gap}, load_solutions=False)

    solve_time = time.time() - iter_start
    is_optimal = result.solver.termination_condition == TerminationCondition.optimal

    # Load solution only if optimal or feasible
    if is_optimal or result.solver.termination_condition == TerminationCondition.feasible:
        if hasattr(solver, 'load_vars'):
            solver.load_vars()  # APPSI interface
        else:
            model.solutions.load_from(result)  # Legacy interface

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
    tolerance = 1e-10  # Tolerance for cleaning numerical noise

    for var_name, values in var_values.items():
        if hasattr(new_model, var_name):
            new_var = getattr(new_model, var_name)
            for idx, val in values.items():
                if idx in new_var:
                    # Clean numerical errors before setting values
                    cleaned_val = val

                    # For binary variables: round to 0 or 1
                    if new_var[idx].domain == Binary:
                        cleaned_val = round(val)
                    # For non-negative variables: ensure non-negativity
                    elif hasattr(new_var[idx].domain, 'bounds'):
                        bounds = new_var[idx].domain.bounds()
                        if bounds[0] == 0:  # NonNegativeReals, UnitInterval, etc.
                            cleaned_val = max(0.0, val)
                            if bounds[1] == 1:  # UnitInterval
                                cleaned_val = min(1.0, cleaned_val)
                    # For general continuous variables with small negative noise
                    elif abs(val) < tolerance:
                        cleaned_val = 0.0

                    # Check and clip against explicit variable bounds (lb, ub)
                    # This handles cases like pw[gen,t] with bounds=(0.0, max_power)
                    var_obj = new_var[idx]
                    if var_obj.lb is not None:
                        # Allow small violations due to floating point errors
                        if cleaned_val < var_obj.lb:
                            if abs(cleaned_val - var_obj.lb) <= tolerance:
                                cleaned_val = var_obj.lb
                    if var_obj.ub is not None:
                        # Allow small violations due to floating point errors
                        if cleaned_val > var_obj.ub:
                            if abs(cleaned_val - var_obj.ub) <= tolerance:
                                cleaned_val = var_obj.ub

                    new_var[idx].value = cleaned_val
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
                        generators_per_iteration=None, generator_sort_function=None,
                        use_limited_horizon=True):
    """
    Solve UC model using Relax-and-Fix approach with adaptive generator-specific lookahead

    Алгоритм:
    1. Разбить временной горизонт на окна (window_size, window_step)
    2. Разбить генераторы на партии (generators_per_iteration)
       - Если generators_per_iteration=None: все генераторы сразу (классический R&F)
       - Генераторы сортируются по максимальной мощности (убывание)
    3. Для каждого временного окна:
       - Если use_limited_horizon=True:
         * Проанализировать состояние КАЖДОГО генератора на границе фиксации
         * Определить индивидуальный lookahead для каждого генератора:
           - Минимум: step + 1 (для ramping constraints)
           - Extended: step + remaining_uptime/downtime (если незавершен min_uptime/downtime)
         * Деактивировать ограничения индивидуально по генераторам
         * Зафиксировать переменные дальних периодов на 0 (с учетом индивидуального lookahead)
       - Для каждой партии генераторов:
         - Переменные текущей партии в текущем окне - бинарные
         - Переменные в окне релаксации - релаксированы [0,1]
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
        data: Original problem data (required for generator analysis and verification)
        model_builder: Function to build model from data (required if verify_solution=True)
        generators_per_iteration: Number of generators per iteration (None = all at once)
        generator_sort_function: Custom function(data) -> list to sort generators (None = by power desc)
        use_limited_horizon: Use adaptive lookahead window (default: True, recommended)

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

    # Режим адаптивного lookahead
    if verbose:
        if use_limited_horizon:
            print(f"  Adaptive lookahead mode: individual per generator (min +1 for ramping)")
        else:
            print(f"  Full horizon mode: all future periods relaxed")

    # Основной цикл Relax-and-Fix (единая логика для всех стратегий)
    for start in range(0, num_periods, window_step):
        end = min(start + window_size, num_periods)
        step = min(start + window_step, num_periods)

        # Последнее окно - фиксируем все до конца
        if end == num_periods:
            step = end

        window_periods = set(time_periods[start:end])
        fix_periods = set(time_periods[start:step])

        # Определить горизонт релаксации и будущие периоды
        if use_limited_horizon:
            # Для ПЕРВОГО окна (start=0): используем полный горизонт для нахождения начального решения
            # Деактивация ограничений и фиксация переменных начинается только со ВТОРОГО окна
            if start == 0:
                lookahead_periods = set()
                lookahead_start = step
                lookahead_end = num_periods
                generator_lookahead = {}
                if verbose:
                    print(f"  Time Window [{start}:{end}], fixing [{start}:{step}] (first window - full horizon)")
            else:
                # Для последующих окон: используем адаптивный lookahead
                # Анализируем состояние на границе (start - 1) и определяем lookahead для периодов >= step
                generator_lookahead = _calculate_generator_specific_lookahead(
                    model, start, step, data, num_periods
                )

                # Определить максимальный lookahead для создания окна релаксации
                max_lookahead = max(generator_lookahead.values())

                # Окно релаксации начинается после текущего бинарного окна
                lookahead_start = step
                lookahead_end = max_lookahead
                lookahead_periods = set(time_periods[lookahead_start:min(lookahead_end, num_periods)])

                # Дальние будущие периоды (за пределами lookahead всех генераторов)
                future_periods = set(time_periods[lookahead_end:]) if lookahead_end < num_periods else set()

                if verbose:
                    print(f"  Time Window [{start}:{end}], fixing [{start}:{step}], "
                          f"lookahead [{lookahead_start}:{lookahead_end}]")

                # Освободить переменные в окне релаксации (если они были зафиксированы на 0)
                if lookahead_periods:
                    _unfix_variables_in_window(model, lookahead_periods, binary_vars, verbose=verbose)

                # Зафиксировать переменные дальних будущих периодов на 0 (селективно)
                if future_periods:
                    _fix_future_variables_to_zero(model, future_periods, binary_vars,
                                                  generator_lookahead, verbose=verbose)

                # Управлять ограничениями (индивидуально по генераторам)
                _manage_constraints_for_window(model, generator_lookahead, num_periods, verbose=verbose)
        else:
            # Полный горизонт для всех окон
            lookahead_start = step
            lookahead_end = num_periods
            generator_lookahead = {}
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
            # В режиме limited horizon: бинарные в window_periods, релаксированные в lookahead_periods
            _set_variable_domains(binary_vars, current_gen_batch, window_periods)

            # 2. Решить подзадачу
            result, solve_time, is_optimal = _solve_subproblem(model, solver_name, gap)

            # 3. Проверить результат и вывести информацию
            termination = result.solver.termination_condition

            if termination == TerminationCondition.infeasible or \
               termination == TerminationCondition.infeasibleOrUnbounded:
                # Subproblem is infeasible - this is a critical error
                error_msg = [
                    f"\n{'='*60}",
                    "INFEASIBLE SUBPROBLEM DETECTED",
                    f"{'='*60}",
                    f"Time window: [{start}:{end}], fixing [{start}:{step}]",
                ]
                if use_limited_horizon and 'lookahead_end' in locals():
                    error_msg.append(f"Lookahead: [{step}:{lookahead_end}]")

                error_msg.extend([
                    f"Generator batch: [{gen_start_idx}:{gen_end_idx}]",
                    f"Termination condition: {termination}",
                    "",
                    "Possible causes:",
                    "1. Previous window fixed variables create conflicts",
                    "2. Lookahead window is too small for min_uptime/min_downtime constraints",
                    "3. Demand cannot be met with current variable fixings",
                    "",
                    "Suggestions:",
                    "- Try larger window_size or window_step",
                    "- Try use_limited_horizon=False for full horizon",
                    "- Check if the original problem is feasible",
                    f"{'='*60}"
                ])

                raise RuntimeError("\n".join(error_msg))

            if verbose:
                status_str = "optimal" if is_optimal else str(termination)
                indent = "    " if generators_per_iteration < num_generators else "  "

                # Only access objective if solution was loaded
                if is_optimal or termination == TerminationCondition.feasible:
                    obj_str = f"obj={value(model.obj):.2f}"
                else:
                    obj_str = "obj=N/A"

                print(f"{indent}Solved in {solve_time:.2f}s, {obj_str}, status={status_str}")

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
