"""
Relax-and-Fix solver for Unit Commitment problem
"""
import time

from pyomo.environ import Binary, UnitInterval, Var, value, Constraint
from pyomo.opt import SolverFactory, TerminationCondition

from .diagnostics import diagnose_infeasibility


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


def _get_long_horizon_generators(data, window_size):
    """
    Найти генераторы с min_uptime или min_downtime > window_size.

    Эти генераторы нельзя корректно обрабатывать в relax-and-fix с ограниченным окном,
    потому что их constraints охватывают несколько окон. Для таких генераторов
    переменные НЕ должны фиксироваться - они должны оставаться бинарными на всём горизонте.

    Args:
        data: dict - исходные данные задачи с ключом 'thermal_generators'
        window_size: int - размер временного окна

    Returns:
        set: Множество имен генераторов с длинным горизонтом
    """
    thermal_gens = data.get("thermal_generators", {})
    long_horizon_gens = set()

    for gen_name, gen_data in thermal_gens.items():
        min_uptime = gen_data.get("time_up_minimum", 0)
        min_downtime = gen_data.get("time_down_minimum", 0)

        # Если min_uptime или min_downtime > window_size, генератор "длинный"
        if min_uptime > window_size or min_downtime > window_size:
            long_horizon_gens.add(gen_name)

    return long_horizon_gens


def _calculate_generator_specific_lookahead(fix_end_period, window_end_period, data, num_periods):
    """
    Определить минимальный lookahead для каждого генератора (+1 период для ramping constraints)

    Lookahead используется ТОЛЬКО для ограничения gen_limit2 (shutdown ramping), которое
    зависит от wg[g, t+1]. Все остальные ограничения (min_uptime/downtime) обеспечиваются
    через backward unfixing.

    ВАЖНО: Lookahead должен покрывать как минимум до конца текущего окна (window_end_period),
    чтобы ограничения были активны для всех периодов в окне.

    Args:
        fix_end_period: int - конец окна фиксации текущей итерации (step)
        window_end_period: int - конец текущего временного окна
        data: dict - исходные данные задачи с ключом 'thermal_generators'
        num_periods: int - общее количество периодов

    Returns:
        dict: {generator_name: required_end_period} для ВСЕХ генераторов
              Lookahead = max(window_end_period, fix_end_period + 1) (минимальный для ramping)
    """
    thermal_gens = data.get("thermal_generators", {})

    # Минимальный lookahead должен покрывать:
    # 1. Весь текущий window (до window_end_period)
    # 2. +1 период после фиксации для ramping constraints (gen_limit2)
    # На последнем окне lookahead = num_periods (не выходим за границы)
    lookahead_end = min(max(window_end_period, fix_end_period + 1), num_periods)

    # Все генераторы получают одинаковый minimal lookahead
    generator_lookahead = {g: lookahead_end for g in thermal_gens.keys()}

    return generator_lookahead


def _manage_constraints_for_window(model, generator_lookahead, num_periods, active_generators,
                                   window_start, backward_unfixed_periods=None,
                                   backward_unfixed_generators=None,
                                   always_active_generators=None, verbose=False):
    """
    Активировать/деактивировать ограничения с учетом партий генераторов и временного окна

    КРИТИЧЕСКАЯ ЛОГИКА ДЛЯ РЕЛАКСАЦИИ:
    - Генераторы в active_generators (текущая партия) - ВСЕ ограничения активны
    - Генераторы НЕ в active_generators (релаксированные) - только базовые ограничения (PWL + demand)
      → динамические ограничения (logical, uptime, downtime, ramping) ДЕАКТИВИРОВАНЫ
      → это позволяет релаксированным переменным ug ∈ [0,1] не нарушать бинарные ограничения

    ВРЕМЕННЫЕ ГРАНИЦЫ:
    - Прошлые периоды (< window_start, кроме backward_unfixed) - ДЕАКТИВИРОВАНЫ (уже зафиксированы)
    - Текущее окно + lookahead (>= window_start, <= max_lookahead) - АКТИВНЫ
    - Дальнее будущее (> max_lookahead) - ДЕАКТИВИРОВАНЫ (зафиксированы на 0)

    Args:
        model: Pyomo ConcreteModel
        generator_lookahead: dict - {generator_name: required_end_period} для ВСЕХ генераторов
        num_periods: int - общее количество периодов
        active_generators: set - генераторы в текущей партии (бинарные переменные)
        window_start: int - начало текущего временного окна (1-based)
        backward_unfixed_periods: set - периоды, разфиксированные backward unfixing (должны быть активны)
        verbose: bool - выводить информацию о деактивации
    """
    if backward_unfixed_periods is None:
        backward_unfixed_periods = set()
    if backward_unfixed_generators is None:
        backward_unfixed_generators = set()
    if always_active_generators is None:
        always_active_generators = set()

    # Определить максимальный период активности среди всех генераторов
    max_lookahead = max(generator_lookahead.values()) if generator_lookahead else num_periods

    # Минимальный период для активации (с учетом backward unfixing)
    min_active_period = min(backward_unfixed_periods) if backward_unfixed_periods else window_start

    # СИСТЕМНЫЕ ограничения (t) - активны в окне [min_active_period : max_lookahead]
    system_constraints = [
        (model.demand, 0),
        (model.reserves, 0),
    ]

    # ДИНАМИЧЕСКИЕ ограничения - активны ТОЛЬКО для генераторов в active_generators
    # Формат: (constraint_object, time_index_position, generator_index_position)
    dynamic_constraints = [
        # Временные ограничения (min uptime/downtime)
        (model.uptime, 1, 0),          # min uptime
        (model.downtime, 1, 0),        # min downtime
        # Рампинг
        (model.ramp_up, 1, 0),         # ramp up limit
        (model.ramp_down, 1, 0),       # ramp down limit
        # startup_allowed indexed (g, s, t)
        (model.startup_allowed, 2, 0),
    ]

    # БАЗОВЫЕ ограничения - активны для ВСЕХ генераторов (в том числе релаксированных)
    # Эти ограничения обеспечивают участие в балансе спроса, резервов и корректную стоимость
    # КРИТИЧЕСКИ ВАЖНО: gen_limit1/gen_limit2 ограничивают rg сверху, что необходимо
    # для корректного вклада в reserves constraint от всех генераторов
    # ВАЖНО: logical и startup_select должны быть активны для ВСЕХ генераторов,
    # иначе solver может установить wg[g,t]=1 для генератора с ug[g,t-1]=0 (уже выключен),
    # что нарушит gen_limit2[g,t-1]
    basic_constraints = [
        (model.mustrun, 1, 0),         # must-run генераторы
        (model.power_select, 1, 0),    # PWL: pg = sum(...)
        (model.cost_select, 1, 0),     # PWL: cg = sum(...)
        (model.on_select, 1, 0),       # PWL: ug = sum(lg)
        (model.gen_limit1, 1, 0),      # power + reserves upper bound (startup)
        (model.gen_limit2, 1, 0),      # power + reserves upper bound (shutdown)
        (model.logical, 1, 0),         # ug[t] - ug[t-1] = vg[t] - wg[t] - связь с предыдущим периодом
        (model.startup_select, 1, 0),  # vg = sum(dg) - связь startup с vg
    ]

    # ПРИМЕЧАНИЕ: gen_limit2[g, t] использует wg[g, t+1], но специальная обработка
    # (boundary unfixing) создавала конфликты с demand constraints, поэтому
    # gen_limit2 обрабатывается как обычный basic_constraint.

    deactivated_count = 0
    activated_count = 0

    # 1. Управление системными ограничениями [min_active_period : max_lookahead]
    # КРИТИЧЕСКИ ВАЖНО: Системные ограничения (demand, reserves) активируются для
    # ВСЕХ периодов, включая backward-unfixed. Это необходимо для поддержания
    # баланса спроса и резервов при изменении переменных в прошлых периодах.
    # Даже если на промежуточных шагах возникает infeasibility, в конце алгоритм
    # должен найти feasible решение.
    for constraint, time_idx_pos in system_constraints:
        for idx in constraint:
            time_period = idx

            # Активны для ВСЕХ периодов: backward-unfixed + текущее окно + lookahead
            in_active_window = (time_period >= min_active_period) and (time_period <= max_lookahead)

            if in_active_window:
                if not constraint[idx].active:
                    constraint[idx].activate()
                    activated_count += 1
            else:
                if constraint[idx].active:
                    constraint[idx].deactivate()
                    deactivated_count += 1

    # 2. Управление ДИНАМИЧЕСКИМИ ограничениями (только для активной партии)
    for constraint, time_idx_pos, gen_idx_pos in dynamic_constraints:
        for idx in constraint:
            if not isinstance(idx, tuple):
                continue

            generator = idx[gen_idx_pos]
            time_period = idx[time_idx_pos]

            # Динамические ограничения активны для генераторов с бинарными переменными
            is_binary_gen = (generator in active_generators) or (generator in backward_unfixed_generators)
            is_always_active = generator in always_active_generators
            required_end = generator_lookahead.get(generator, max_lookahead)

            # Проверка временного окна:
            # 1. В текущем окне [window_start : required_end]
            # 2. В backward_unfixed периодах - для ВСЕХ генераторов (continuous переменные свободны)
            # 3. Для always_active генераторов - ВСЕ периоды (предотвращает unbounded cg)
            in_current_window = (window_start <= time_period <= required_end)
            in_backward_unfixed = (time_period in backward_unfixed_periods)
            is_always_active_period = is_always_active and (1 <= time_period <= num_periods)
            in_time_window = in_current_window or in_backward_unfixed or is_always_active_period

            should_activate = (is_binary_gen or is_always_active or in_backward_unfixed) and in_time_window

            if should_activate:
                if not constraint[idx].active:
                    constraint[idx].activate()
                    activated_count += 1
            else:
                if constraint[idx].active:
                    constraint[idx].deactivate()
                    deactivated_count += 1

    # 3. Управление БАЗОВЫМИ ограничениями (для всех генераторов в окне)
    for constraint, time_idx_pos, gen_idx_pos in basic_constraints:
        for idx in constraint:
            if not isinstance(idx, tuple):
                continue

            generator = idx[gen_idx_pos]
            time_period = idx[time_idx_pos]

            is_always_active = generator in always_active_generators

            # Базовые ограничения активны для ВСЕХ генераторов:
            # 1. В текущем окне [window_start : required_end]
            # 2. В backward_unfixed периодах - для ВСЕХ генераторов (continuous переменные свободны)
            # 3. Для always_active генераторов - ВСЕ периоды (предотвращает unbounded cg)
            required_end = generator_lookahead.get(generator, max_lookahead)
            in_current_window = (window_start <= time_period <= required_end)
            in_backward_unfixed = (time_period in backward_unfixed_periods)
            is_always_active_period = is_always_active and (1 <= time_period <= num_periods)
            should_activate = in_current_window or in_backward_unfixed or is_always_active_period

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
        print(f"    Active window: [{min_active_period}:{max_lookahead}]")
        print(f"    Active generators (binary): {len(active_generators)}")
        if backward_unfixed_periods:
            print(f"    Backward unfixed periods: {sorted(backward_unfixed_periods)}")


def _fix_future_variables_to_zero(model, future_periods, binary_vars, generator_lookahead,
                                   always_binary_generators=None, verbose=False):
    """
    Селективно зафиксировать переменные для дальних будущих периодов на 0

    НЕ фиксирует переменные:
    - Критических генераторов в их extended lookahead периодах
    - Генераторов из always_binary_generators (никогда не фиксируются)

    Args:
        model: Pyomo ConcreteModel
        future_periods: set - периоды вне базового горизонта релаксации
        binary_vars: list - список (var, time_idx_pos) бинарных переменных
        generator_lookahead: dict - {generator_name: required_end_period} для критических генераторов
        always_binary_generators: set - генераторы, которые НИКОГДА не фиксируются
        verbose: bool - выводить информацию

    Note:
        Также фиксирует непрерывные переменные (pg, rg, cg) на 0 для этих периодов.
    """
    if always_binary_generators is None:
        always_binary_generators = set()

    fixed_count = 0
    skipped_count = 0

    # Фиксируем бинарные переменные (ug, vg, wg, dg)
    for var, time_idx_pos in binary_vars:
        for idx in var:
            generator = idx[0]  # Генератор всегда на позиции 0
            time_period = idx[time_idx_pos]

            if time_period not in future_periods or var[idx].is_fixed():
                continue

            # Генераторы из always_binary_generators НИКОГДА не фиксируются
            if generator in always_binary_generators:
                skipped_count += 1
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

            # Генераторы из always_binary_generators НИКОГДА не фиксируются
            if generator in always_binary_generators:
                skipped_count += 1
                continue

            # Проверить extended lookahead
            if generator in generator_lookahead:
                if time_period <= generator_lookahead[generator]:
                    skipped_count += 1
                    continue

            var[idx].fix(0)
            fixed_count += 1

    # Фиксируем renewable generation (pw) на нижнюю границу для будущих периодов
    # ВАЖНО: pw не привязана к demand (demand деактивирован для будущих периодов),
    # поэтому нужно зафиксировать, иначе при загрузке решения значения будут некорректны
    if hasattr(model, 'pw'):
        for idx in model.pw:
            time_period = idx[1]  # (w, t)
            if time_period not in future_periods or model.pw[idx].is_fixed():
                continue
            # Фиксируем на нижнюю границу (минимальная renewable generation)
            lb = model.pw[idx].lb
            model.pw[idx].fix(lb if lb is not None else 0)
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

    # Освобождаем непрерывные переменные (thermal)
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

    # Освобождаем переменные renewable generation (pw)
    if hasattr(model, 'pw'):
        for idx in model.pw:
            time_period = idx[1]  # (w, t)
            if time_period in window_periods and model.pw[idx].is_fixed():
                model.pw[idx].unfix()
                unfixed_count += 1

    if verbose and unfixed_count > 0:
        print(f"    Unfixed {unfixed_count} variables in window")


def _unfix_boundary_period_for_gen_limit2(model, boundary_period, verbose=False):
    """
    Разфиксировать переменные pg, rg, lg, cg для граничного периода.

    Это необходимо для constraint gen_limit2[g, t], который использует wg[g, t+1].
    Когда wg[g, window_start] решается как бинарная, constraint gen_limit2[g, window_start-1]
    должен быть активен, и переменные pg, rg должны быть свободны для изменения.

    ВАЖНО: Разфиксируем ТОЛЬКО для генераторов с ug[g, boundary_period] = 1 (зафиксировано).
    Эти генераторы могут выключиться в window_start, и constraint gen_limit2 ограничивает их.
    Для генераторов с ug=0 constraint gen_limit2 всё равно требует pg+rg <= 0, что и так выполняется.

    Args:
        model: Pyomo ConcreteModel
        boundary_period: int - период для разфиксации (обычно window_start - 1)
        verbose: bool - выводить информацию

    Returns:
        int: количество разфиксированных переменных
    """
    from pyomo.environ import value

    if boundary_period < 1:
        return 0

    unfixed_count = 0
    unfixed_generators = set()

    # Определить генераторы, которые были включены в boundary_period (могут выключиться)
    generators_on = set()
    for idx in model.ug:
        gen_name, time_period = idx
        if time_period == boundary_period and model.ug[idx].is_fixed():
            ug_val = value(model.ug[idx])
            if abs(ug_val - 1.0) < 1e-6:  # ug = 1 (включен)
                generators_on.add(gen_name)

    if not generators_on:
        return 0

    # Разфиксировать переменные ТОЛЬКО для генераторов, которые были включены
    continuous_vars = [
        (model.pg, 1),  # (g, t)
        (model.rg, 1),  # (g, t)
        (model.cg, 1),  # (g, t)
    ]

    for var, time_idx_pos in continuous_vars:
        for idx in var:
            gen_name = idx[0]
            time_period = idx[time_idx_pos]
            if (time_period == boundary_period and
                gen_name in generators_on and
                var[idx].is_fixed()):
                var[idx].unfix()
                unfixed_count += 1
                unfixed_generators.add(gen_name)

    # lg имеет структуру (g, l, t) - время на позиции 2
    if hasattr(model, 'lg'):
        for idx in model.lg:
            gen_name = idx[0]
            time_period = idx[2]  # (g, l, t)
            if (time_period == boundary_period and
                gen_name in generators_on and
                model.lg[idx].is_fixed()):
                model.lg[idx].unfix()
                unfixed_count += 1
                unfixed_generators.add(gen_name)

    if verbose and unfixed_count > 0:
        print(f"    Unfixed {unfixed_count} boundary variables for period {boundary_period} "
              f"({len(unfixed_generators)} generators with ug=1)")

    return unfixed_count


def _refix_boundary_period(model, boundary_period, verbose=False):
    """
    Зафиксировать обратно переменные граничного периода после решения.

    Фиксирует только переменные, которые были разфиксированы (не is_fixed()).

    Args:
        model: Pyomo ConcreteModel
        boundary_period: int - период для фиксации
        verbose: bool - выводить информацию

    Returns:
        int: количество зафиксированных переменных
    """
    if boundary_period < 1:
        return 0

    from pyomo.environ import value

    fixed_count = 0

    continuous_vars = [
        (model.pg, 1),
        (model.rg, 1),
        (model.cg, 1),
    ]

    for var, time_idx_pos in continuous_vars:
        for idx in var:
            time_period = idx[time_idx_pos]
            if time_period == boundary_period and not var[idx].is_fixed():
                current_value = value(var[idx])
                # Clip для избежания численных ошибок
                if current_value < 0 and abs(current_value) < 1e-10:
                    current_value = 0.0
                var[idx].value = current_value
                var[idx].fix()
                fixed_count += 1

    # lg
    if hasattr(model, 'lg'):
        for idx in model.lg:
            time_period = idx[2]
            if time_period == boundary_period and not model.lg[idx].is_fixed():
                current_value = value(model.lg[idx])
                current_value = max(0.0, min(1.0, current_value))
                model.lg[idx].value = current_value
                model.lg[idx].fix()
                fixed_count += 1

    if verbose and fixed_count > 0:
        print(f"    Re-fixed {fixed_count} boundary variables for period {boundary_period}")

    return fixed_count


def _set_variable_domains(binary_vars, binary_generators, binary_periods,
                          backward_unfixed_generators=None, backward_unfixed_periods=None,
                          always_binary_generators=None,
                          verbose=False):
    """
    Установить домены переменных для текущей подзадачи

    Args:
        binary_vars: список (var, time_idx_pos) - бинарные переменные модели
        binary_generators: set - генераторы, которые должны быть бинарными
        binary_periods: set - временные периоды, которые должны быть бинарными
        backward_unfixed_generators: set - генераторы, разфиксированные backward unfixing
        backward_unfixed_periods: set - периоды, разфиксированные backward unfixing
        always_binary_generators: set - генераторы, которые ВСЕГДА бинарные (на всём горизонте)
        verbose: bool - выводить информацию
    """
    if backward_unfixed_generators is None:
        backward_unfixed_generators = set()
    if backward_unfixed_periods is None:
        backward_unfixed_periods = set()
    if always_binary_generators is None:
        always_binary_generators = set()

    binary_count = 0
    relaxed_count = 0
    backward_binary_count = 0
    always_binary_count = 0

    for var, time_idx_pos in binary_vars:
        for idx in var:
            time_period = idx[time_idx_pos]
            generator = idx[0]  # Для всех переменных UC модели генератор на позиции 0

            # Бинарные: (генератор в партии И период в окне) ИЛИ (генератор разфиксирован И период разфиксирован)
            # ИЛИ (генератор в always_binary - никогда не релаксируется)
            is_current_batch = generator in binary_generators and time_period in binary_periods
            is_backward_unfixed = generator in backward_unfixed_generators and time_period in backward_unfixed_periods
            is_always_binary = generator in always_binary_generators

            if is_current_batch or is_backward_unfixed or is_always_binary:
                if not var[idx].is_fixed():
                    var[idx].domain = Binary
                    binary_count += 1
                    if is_backward_unfixed:
                        backward_binary_count += 1
                    if is_always_binary and not is_current_batch and not is_backward_unfixed:
                        always_binary_count += 1
            elif not var[idx].is_fixed():
                var[idx].domain = UnitInterval
                relaxed_count += 1

    if verbose and (binary_count > 0 or relaxed_count > 0):
        extra_info = f", {always_binary_count} always-binary" if always_binary_count > 0 else ""
        print(f"    Domains: {binary_count} Binary ({backward_binary_count} backward-unfixed{extra_info}), {relaxed_count} Relaxed")


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

        try:
            result = solver.solve(model)
        except RuntimeError as e:
            # HiGHS может бросить RuntimeError если решение не найдено
            # В этом случае все равно возвращаем результат с termination_condition
            if "feasible solution was not found" in str(e):
                # Для APPSI solvers, результат уже в solver после вызова solve
                # Создаем mock результат для обратной совместимости
                from pyomo.opt import SolverResults
                result = SolverResults()
                result.solver.termination_condition = TerminationCondition.infeasible
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


def _fix_variables(binary_vars, fix_generators, fix_periods, model=None):
    """
    Зафиксировать переменные для текущей подзадачи

    КРИТИЧЕСКИ ВАЖНО: Фиксирует НЕ ТОЛЬКО бинарные переменные, но и все связанные
    continuous переменные (pg, rg, cg, lg), чтобы они не входили в objective без ограничений!

    Args:
        binary_vars: список (var, time_idx_pos) - бинарные переменные модели
        fix_generators: set - генераторы, которые нужно зафиксировать
        fix_periods: set - периоды, которые нужно зафиксировать
        model: Pyomo ConcreteModel - нужен для доступа к pg, rg, cg, lg
    """
    # 1. Фиксация бинарных переменных
    for var, time_idx_pos in binary_vars:
        for idx in var:
            generator = idx[0]  # Генератор на первой позиции
            time_period = idx[time_idx_pos]

            # Фиксируем: генератор в партии И период в окне фиксации И не зафиксирован
            if (generator in fix_generators and
                    time_period in fix_periods and
                    not var[idx].is_fixed()):
                # Бинарные переменные решались как Binary (не Relaxed),
                # поэтому должны быть точно 0 или 1. Только очистка численных ошибок.
                current_value = value(var[idx])

                # Очистка численных ошибок (например, 0.9999999 -> 1.0, 0.0000001 -> 0.0)
                if abs(current_value - 1.0) < 1e-6:
                    current_value = 1.0
                elif abs(current_value) < 1e-6:
                    current_value = 0.0

                var[idx].value = current_value
                var[idx].fix()

    # 2. КРИТИЧЕСКИ ВАЖНО: Фиксация continuous переменных (pg, rg, cg, lg)
    # Эти переменные входят в objective, поэтому ДОЛЖНЫ быть зафиксированы!
    # НЕ обнуляем их для "выключенных" генераторов, чтобы не нарушить demand constraints
    if model is not None:
        continuous_vars = ['pg', 'rg', 'cg', 'lg']

        for var_name in continuous_vars:
            if not hasattr(model, var_name):
                continue

            var_obj = getattr(model, var_name)

            for idx in var_obj:
                generator = idx[0]  # Генератор на первой позиции

                # Определить позицию времени
                if len(idx) == 2:  # (g, t)
                    time_period = idx[1]
                elif len(idx) == 3:  # (g, l, t)
                    time_period = idx[2]
                else:
                    continue

                # Фиксируем: генератор в партии И период в окне фиксации И не зафиксирован
                if (generator in fix_generators and
                        time_period in fix_periods and
                        not var_obj[idx].is_fixed()):
                    current_value = value(var_obj[idx])

                    # ВАЖНО: Clip значения для избежания численных ошибок
                    # lg должны быть в [0, 1], остальные могут быть >= 0
                    if var_name == 'lg':
                        # Clip to [0, 1] с небольшим tolerance
                        current_value = max(0.0, min(1.0, current_value))
                    elif current_value < 0 and abs(current_value) < 1e-10:
                        # Маленькие отрицательные значения (~0) -> 0
                        current_value = 0.0

                    var_obj[idx].value = current_value
                    var_obj[idx].fix()

        # 3. КРИТИЧЕСКИ ВАЖНО: Фиксация renewable generation (pw)
        # Renewable variables не зависят от партий генераторов, фиксируем для всех renewables
        if hasattr(model, 'pw'):
            for idx in model.pw:
                # idx = (renewable_gen, time_period)
                time_period = idx[1]

                if time_period in fix_periods and not model.pw[idx].is_fixed():
                    current_value = value(model.pw[idx])

                    # Clip to variable bounds to avoid floating point errors
                    var_obj = model.pw[idx]
                    if var_obj.lb is not None and current_value < var_obj.lb:
                        if abs(current_value - var_obj.lb) <= 1e-10:
                            current_value = var_obj.lb
                    if var_obj.ub is not None and current_value > var_obj.ub:
                        if abs(current_value - var_obj.ub) <= 1e-10:
                            current_value = var_obj.ub

                    model.pw[idx].value = current_value
                    model.pw[idx].fix()


def _refix_periods(model, periods_to_fix, verbose=False):
    """
    Зафиксировать все переменные в указанных периодах на их текущих значениях

    Args:
        model: Pyomo ConcreteModel
        periods_to_fix: set - периоды для фиксации
        verbose: bool - выводить информацию
    """
    if not periods_to_fix:
        return

    fixed_count = 0

    # Фиксируем все бинарные переменные в указанных периодах
    for var_name in ['ug', 'vg', 'wg', 'dg']:
        if not hasattr(model, var_name):
            continue

        var_obj = getattr(model, var_name)

        for idx in var_obj:
            # Определить позицию времени
            if len(idx) == 2:  # (g, t)
                time_period = idx[1]
            elif len(idx) == 3:  # (g, s, t) или (g, l, t)
                time_period = idx[2]
            else:
                continue

            if time_period in periods_to_fix and not var_obj[idx].is_fixed():
                # Бинарные переменные в backward-unfixed периодах решались как Binary (не Relaxed),
                # поэтому должны быть точно 0 или 1. Только очистка численных ошибок.
                current_value = value(var_obj[idx])

                # Очистка численных ошибок (например, 0.9999999 -> 1.0, 0.0000001 -> 0.0)
                if abs(current_value - 1.0) < 1e-6:
                    current_value = 1.0
                elif abs(current_value) < 1e-6:
                    current_value = 0.0

                var_obj[idx].value = current_value
                var_obj[idx].fix()
                fixed_count += 1

    # КРИТИЧЕСКИ ВАЖНО: Фиксируем continuous переменные (pg, rg, cg, lg, pw)
    # НЕ обнуляем их для "выключенных" генераторов, чтобы не нарушить demand constraints
    # ВАЖНО: Включаем pw (renewable generation) для поддержания баланса спроса
    for var_name in ['pg', 'rg', 'cg', 'lg', 'pw']:
        if not hasattr(model, var_name):
            continue

        var_obj = getattr(model, var_name)

        for idx in var_obj:
            # Определить позицию времени
            if len(idx) == 2:  # (g, t) или (w, t)
                time_period = idx[1]
            elif len(idx) == 3:  # (g, l, t)
                time_period = idx[2]
            else:
                continue

            if time_period in periods_to_fix and not var_obj[idx].is_fixed():
                current_value = value(var_obj[idx])

                # ВАЖНО: Clip значения для избежания численных ошибок
                if var_name == 'lg':
                    current_value = max(0.0, min(1.0, current_value))
                elif var_name == 'pw':
                    # Clip to variable bounds to avoid floating point errors
                    var_instance = var_obj[idx]
                    if var_instance.lb is not None and current_value < var_instance.lb:
                        if abs(current_value - var_instance.lb) <= 1e-10:
                            current_value = var_instance.lb
                    if var_instance.ub is not None and current_value > var_instance.ub:
                        if abs(current_value - var_instance.ub) <= 1e-10:
                            current_value = var_instance.ub
                elif current_value < 0 and abs(current_value) < 1e-10:
                    current_value = 0.0

                var_obj[idx].value = current_value
                var_obj[idx].fix()
                fixed_count += 1

    if verbose and fixed_count > 0:
        print(f"    Re-fixed {fixed_count} variables in backward-unfixed periods")


def _unfix_gen_vars_in_range(model, gen_name, gen_data, unfix_start, unfix_end, unfixed_periods):
    """
    Разфиксировать все переменные одного генератора в заданном диапазоне периодов.

    Использует прямой доступ по индексам вместо полного обхода model.dg/model.lg.

    Returns:
        int: количество разфиксированных переменных
    """
    count = 0
    startup_cats = list(range(len(gen_data.get("startup", []))))
    pwl_points = list(range(len(gen_data.get("piecewise_production", []))))

    for t in range(unfix_start, unfix_end + 1):
        period_unfixed = False

        # Бинарные: ug, vg, wg
        for var_name in ['ug', 'vg', 'wg']:
            if hasattr(model, var_name):
                idx = (gen_name, t)
                var_obj = getattr(model, var_name)
                if idx in var_obj and var_obj[idx].is_fixed():
                    var_obj[idx].unfix()
                    count += 1
                    period_unfixed = True

        # Добавляем период только если были разфиксированы бинарные переменные
        if period_unfixed:
            unfixed_periods.add(t)

        # Continuous: pg, rg, cg
        for var_name in ['pg', 'rg', 'cg']:
            if hasattr(model, var_name):
                idx = (gen_name, t)
                var_obj = getattr(model, var_name)
                if idx in var_obj and var_obj[idx].is_fixed():
                    var_obj[idx].unfix()
                    count += 1

        # dg: (generator, startup_cat, time)
        if hasattr(model, 'dg'):
            for s in startup_cats:
                idx = (gen_name, s, t)
                if idx in model.dg and model.dg[idx].is_fixed():
                    model.dg[idx].unfix()
                    count += 1

        # lg: (generator, pwl_point, time)
        if hasattr(model, 'lg'):
            for l in pwl_points:
                idx = (gen_name, l, t)
                if idx in model.lg and model.lg[idx].is_fixed():
                    model.lg[idx].unfix()
                    count += 1

    return count


def _backward_unfix_for_startup(model, window_start, data, verbose=False,
                                  allowed_generators=None):
    """
    Разфиксировать минимальное количество переменных из прошлых окон для обеспечения достижимости

    КЛЮЧЕВАЯ ЛОГИКА ДЛЯ ПРЕДОТВРАЩЕНИЯ НЕДОПУСТИМОСТИ:
    Lookahead минимальный (+1 для ramping constraints). Backward unfixing компенсирует
    отсутствие большого lookahead, разрешая изменить прошлые решения для обеспечения допустимости:

    1. Генератор OFF на границе + min_downtime не выполнен → не может включиться в новом окне
       → Разфиксируем прошлые периоды, чтобы дать возможность "досидеть" downtime

    2. Генератор ON на границе + min_uptime не выполнен → не может выключиться в новом окне
       → Разфиксируем прошлые периоды, чтобы дать возможность "доработать" uptime

    3. Multi-stage startup: учитываем максимальный startup lag

    ВАЖНО: Разфиксируются ВСЕ переменные генератора (не только бинарные ug, vg, wg, dg,
    но и continuous pg, rg, cg, lg), чтобы избежать конфликтов между зафиксированными
    continuous переменными и изменившимися бинарными.

    КРИТИЧЕСКИ ВАЖНО: allowed_generators ограничивает backward unfixing только генераторами
    текущей партии. Это предотвращает конфликт, когда переменные генератора из предыдущей
    партии разфиксируются в прошлых периодах, но уже зафиксированы в текущем окне.

    Расчёт точен и детерминирован - недопустимость исключена при корректных данных.

    Args:
        model: Pyomo ConcreteModel
        window_start: int - начало нового окна (1-based)
        data: dict - исходные данные с thermal_generators
        verbose: bool - выводить информацию
        allowed_generators: set - если задано, разфиксировать только эти генераторы

    Returns:
        tuple: (unfixed_periods: set, unfixed_generators: set) - periods and generators that were unfixed
    """
    if window_start <= 1:
        return set(), set()

    thermal_gens = data.get("thermal_generators", {})
    unfixed_periods = set()
    unfixed_generators = set()
    unfixed_count = 0

    for gen_name, gen_data in thermal_gens.items():
        # Пропустить генераторы, не входящие в разрешённый список
        if allowed_generators is not None and gen_name not in allowed_generators:
            continue
        min_downtime = gen_data.get("time_down_minimum", 0)
        min_uptime = gen_data.get("time_up_minimum", 0)

        # Учитываем startup lags для multi-stage startup
        startup_categories = gen_data.get("startup", [])
        max_startup_lag = max([s.get("lag", 0) for s in startup_categories], default=0)

        boundary_period = window_start - 1

        if (gen_name, boundary_period) not in model.ug:
            continue

        ug_boundary = model.ug[gen_name, boundary_period]

        if not ug_boundary.is_fixed():
            continue

        ug_boundary_value = round(value(ug_boundary))

        # СЛУЧАЙ 1: Генератор OFF на границе, может потребоваться включить в новом окне
        # Ограничение: должен быть выключен min_downtime периодов подряд
        if ug_boundary_value == 0 and (min_downtime > 0 or max_startup_lag > 0):
            # Посчитать сколько периодов подряд он был OFF
            consecutive_off = 1
            for t in range(boundary_period - 1, 0, -1):
                if (gen_name, t) in model.ug and model.ug[gen_name, t].is_fixed():
                    if round(value(model.ug[gen_name, t])) == 0:
                        consecutive_off += 1
                    else:
                        break
                else:
                    break

            # Определить необходимую глубину unfixing
            # Учитываем как min_downtime, так и startup lags
            required_downtime = max(min_downtime, max_startup_lag)

            # Если не выполнен требуемый downtime, разфиксировать переменные
            if consecutive_off < required_downtime:
                # Разфиксировать окно, достаточное для выполнения ограничений
                # От (window_start - required_downtime) до (boundary_period)
                unfix_start = max(1, window_start - required_downtime)
                unfix_end = boundary_period

                unfixed_generators.add(gen_name)

                for t in range(unfix_start, unfix_end + 1):
                    # Бинарные: ug, vg, wg
                    for var_name in ['ug', 'vg', 'wg']:
                        idx = (gen_name, t)
                        var_obj = getattr(model, var_name)
                        if idx in var_obj and var_obj[idx].is_fixed():
                            var_obj[idx].unfix()
                            unfixed_count += 1
                            unfixed_periods.add(t)
                    # Continuous: pg, rg, cg
                    for var_name in ['pg', 'rg', 'cg']:
                        idx = (gen_name, t)
                        var_obj = getattr(model, var_name)
                        if idx in var_obj and var_obj[idx].is_fixed():
                            var_obj[idx].unfix()
                            unfixed_count += 1
                    # dg: iterate all indices
                    for idx in model.dg:
                        g, s, tp = idx
                        if g == gen_name and tp == t and model.dg[idx].is_fixed():
                            model.dg[idx].unfix()
                            unfixed_count += 1
                    # lg: iterate all indices
                    for idx in model.lg:
                        g, l, tp = idx
                        if g == gen_name and tp == t and model.lg[idx].is_fixed():
                            model.lg[idx].unfix()
                            unfixed_count += 1

        # СЛУЧАЙ 2: Генератор ON на границе, может потребоваться выключить в новом окне
        # Ограничение: должен быть включен min_uptime периодов подряд
        elif ug_boundary_value == 1 and min_uptime > 0:
            # Посчитать сколько периодов подряд он был ON
            consecutive_on = 1
            for t in range(boundary_period - 1, 0, -1):
                if (gen_name, t) in model.ug and model.ug[gen_name, t].is_fixed():
                    if round(value(model.ug[gen_name, t])) == 1:
                        consecutive_on += 1
                    else:
                        break
                else:
                    break

            # Если не выполнен min_uptime, разфиксировать переменные
            if consecutive_on < min_uptime:
                # Разфиксировать окно от (window_start - min_uptime) до (boundary_period)
                unfix_start = max(1, window_start - min_uptime)
                unfix_end = boundary_period

                unfixed_generators.add(gen_name)

                for t in range(unfix_start, unfix_end + 1):
                    # Бинарные: ug, vg, wg
                    for var_name in ['ug', 'vg', 'wg']:
                        idx = (gen_name, t)
                        var_obj = getattr(model, var_name)
                        if idx in var_obj and var_obj[idx].is_fixed():
                            var_obj[idx].unfix()
                            unfixed_count += 1
                            unfixed_periods.add(t)
                    # Continuous: pg, rg, cg
                    for var_name in ['pg', 'rg', 'cg']:
                        idx = (gen_name, t)
                        var_obj = getattr(model, var_name)
                        if idx in var_obj and var_obj[idx].is_fixed():
                            var_obj[idx].unfix()
                            unfixed_count += 1
                    # dg: iterate all indices
                    for idx in model.dg:
                        g, s, tp = idx
                        if g == gen_name and tp == t and model.dg[idx].is_fixed():
                            model.dg[idx].unfix()
                            unfixed_count += 1
                    # lg: iterate all indices
                    for idx in model.lg:
                        g, l, tp = idx
                        if g == gen_name and tp == t and model.lg[idx].is_fixed():
                            model.lg[idx].unfix()
                            unfixed_count += 1

    # КРИТИЧЕСКИ ВАЖНО: Разфиксировать continuous переменные для ВСЕХ генераторов
    # в backward-unfixed периодах. Это необходимо для перебалансировки demand:
    # когда целевые генераторы меняют ON/OFF статус, остальные генераторы с ug=1
    # должны иметь возможность скорректировать pg для поддержания баланса спроса.
    if unfixed_periods:
        all_continuous_unfixed = 0

        # Эффективный обход: итерация по ВСЕМ индексам переменной один раз
        # вместо вложенных циклов по генераторам × периодам

        # pg, rg, cg: (generator, time) — один проход по каждой переменной
        for var_name in ['pg', 'rg', 'cg']:
            if not hasattr(model, var_name):
                continue
            var_obj = getattr(model, var_name)
            for idx in var_obj:
                gen_name, t = idx
                if gen_name in unfixed_generators:
                    continue  # Уже разфиксированы выше
                if t in unfixed_periods and var_obj[idx].is_fixed():
                    var_obj[idx].unfix()
                    all_continuous_unfixed += 1

        # lg: (generator, pwl_point, time) — один проход
        if hasattr(model, 'lg'):
            for idx in model.lg:
                gen_name, _, t = idx
                if gen_name in unfixed_generators:
                    continue
                if t in unfixed_periods and model.lg[idx].is_fixed():
                    model.lg[idx].unfix()
                    all_continuous_unfixed += 1

        # Renewable generators: unfix pw для ВСЕХ renewables — один проход
        if hasattr(model, 'pw'):
            for idx in model.pw:
                _, t = idx
                if t in unfixed_periods and model.pw[idx].is_fixed():
                    model.pw[idx].unfix()
                    all_continuous_unfixed += 1

        unfixed_count += all_continuous_unfixed

        if verbose and all_continuous_unfixed > 0:
            print(f"    Demand rebalancing: unfixed {all_continuous_unfixed} continuous vars "
                  f"for all generators in backward periods")

    if verbose and unfixed_count > 0:
        unique_periods = sorted(unfixed_periods)
        print(f"    Backward unfixing: {unfixed_count} total variables in {len(unique_periods)} periods "
              f"({min(unique_periods)}-{max(unique_periods)})")
        if len(unfixed_generators) <= 10:
            print(f"    Unfixed generators (binary): {', '.join(sorted(unfixed_generators))}")
        else:
            print(f"    Unfixed generators (binary): {len(unfixed_generators)} total")

    return unfixed_periods, unfixed_generators


def _final_lp_reoptimization(model, data, model_builder, solver_name, gap, verbose):
    """
    Финальная LP re-optimization: зафиксировать бинарные переменные, освободить continuous,
    активировать ВСЕ ограничения и решить LP.

    Это стандартная техника для R&F, которая гарантирует допустимость continuous переменных
    при данных бинарных решениях.

    Args:
        model: Pyomo ConcreteModel после R&F (с зафиксированными переменными)
        data: Исходные данные задачи
        model_builder: Функция построения модели
        solver_name: Имя солвера
        gap: MIP gap
        verbose: Подробный вывод

    Returns:
        dict: {'success': bool, 'objective': float, 'model': ConcreteModel}
    """
    if verbose:
        print("\n  Final LP re-optimization (fixing binary decisions, freeing continuous)...")

    # Собрать бинарные решения из R&F
    binary_decisions = {}
    for var_name in ['ug', 'vg', 'wg', 'dg']:
        if not hasattr(model, var_name):
            continue
        var_obj = getattr(model, var_name)
        binary_decisions[var_name] = {}
        for idx in var_obj:
            val = value(var_obj[idx])
            # Округляем до 0/1
            if abs(val - 1.0) < 1e-6:
                val = 1.0
            elif abs(val) < 1e-6:
                val = 0.0
            else:
                val = round(val)
            binary_decisions[var_name][idx] = val

    # Построить свежую модель
    fresh_model = model_builder(data)

    # Зафиксировать бинарные переменные
    for var_name, decisions in binary_decisions.items():
        var_obj = getattr(fresh_model, var_name)
        for idx, val in decisions.items():
            if idx in var_obj:
                var_obj[idx].fix(val)

    # Все continuous переменные и ограничения остаются свободными/активными
    # Решить LP
    solver = SolverFactory(solver_name)
    if hasattr(solver, 'config'):
        solver.config.mip_gap = gap
        try:
            result = solver.solve(fresh_model)
        except RuntimeError as e:
            if "feasible solution was not found" in str(e):
                if verbose:
                    print("  LP re-optimization FAILED: infeasible with given binary decisions")
                return {'success': False, 'objective': None, 'model': fresh_model}
            raise
    else:
        result = solver.solve(fresh_model, options={'ratioGap': gap})

    is_ok = result.solver.termination_condition in (
        TerminationCondition.optimal, TerminationCondition.feasible
    )

    if is_ok:
        if hasattr(solver, 'load_vars'):
            solver.load_vars()
        obj = value(fresh_model.obj)
        if verbose:
            print(f"  LP re-optimization SUCCESS: obj={obj:.2f}")
        return {'success': True, 'objective': obj, 'model': fresh_model}
    else:
        if verbose:
            print(f"  LP re-optimization FAILED: {result.solver.termination_condition}")
        return {'success': False, 'objective': None, 'model': fresh_model}


def _verify_solution_feasibility(old_model, data, model_builder, solver_name, gap, verbose):
    """
    Проверить допустимость найденного решения в исходной (немодифицированной) задаче

    Создаёт НОВУЮ модель из исходных данных (без модификаций),
    фиксирует ТОЛЬКО бинарные переменные из решения и решает LP.
    Это избегает накопления численных ошибок при копировании continuous переменных.

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

    # Извлекаем только бинарные решения из старой модели
    binary_decisions = {}
    for var_name in ['ug', 'vg', 'wg', 'dg']:
        if not hasattr(old_model, var_name):
            continue
        var_obj = getattr(old_model, var_name)
        binary_decisions[var_name] = {}
        for idx in var_obj:
            val = value(var_obj[idx])
            # Округляем до 0/1
            if abs(val - 1.0) < 1e-6:
                val = 1.0
            elif abs(val) < 1e-6:
                val = 0.0
            else:
                val = round(val)
            binary_decisions[var_name][idx] = val

    # Создать НОВУЮ модель из исходных данных
    if verbose:
        print("    Building fresh model from original data...")

    new_model = model_builder(data)

    # Зафиксировать ТОЛЬКО бинарные переменные
    if verbose:
        print("    Fixing binary variables in fresh model...")

    for var_name, decisions in binary_decisions.items():
        var_obj = getattr(new_model, var_name)
        for idx, val in decisions.items():
            if idx in var_obj:
                var_obj[idx].fix(val)

    # Решить LP (все continuous переменные свободны, все ограничения активны)
    if verbose:
        print("    Solving LP with fixed binary decisions...")

    solver = SolverFactory(solver_name)

    if hasattr(solver, 'config'):
        solver.config.mip_gap = gap
        try:
            result = solver.solve(new_model, tee=False)
        except RuntimeError as e:
            if "feasible solution was not found" in str(e):
                from pyomo.opt import SolverResults
                result = SolverResults()
                result.solver.termination_condition = TerminationCondition.infeasible
            else:
                raise
    else:
        result = solver.solve(new_model, options={'ratioGap': gap}, tee=False)

    # Проверить результаты
    is_feasible = result.solver.termination_condition in (
        TerminationCondition.optimal, TerminationCondition.feasible
    )

    if is_feasible:
        if hasattr(solver, 'load_vars'):
            solver.load_vars()
        verified_objective = value(new_model.obj)
    else:
        verified_objective = None

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
            print(f"    The binary decisions from R&F do not admit a feasible continuous dispatch")

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
    Solve UC model using Relax-and-Fix approach with minimal lookahead and backward unfixing

    Алгоритм:
    1. Разбить временной горизонт на окна (window_size, window_step)
    2. Разбить генераторы на партии (generators_per_iteration)
       - Если generators_per_iteration=None: все генераторы сразу (классический R&F)
       - Генераторы сортируются по максимальной мощности (убывание)
    3. Для каждого временного окна:
       - Backward unfixing: разфиксировать минимальное количество переменных из прошлых окон
         для обеспечения min_uptime/min_downtime constraints
       - Если use_limited_horizon=True:
         * Lookahead минимальный: step + 1 (для ramping constraints)
         * Зафиксировать переменные дальних периодов на 0
       - Для каждой партии генераторов:
         * Переменные текущей партии в текущем окне - бинарные
         * Переменные в окне релаксации - релаксированы [0,1]
         * Решить подзадачу
         * Зафиксировать переменные партии в окне фиксации
    4. Двигаем окно дальше

    ФИЛОСОФИЯ: Lookahead минимален (+1 для ramping), а вся сложность min_uptime/downtime
    обрабатывается через backward unfixing (просмотр назад, а не вперёд).

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
        use_limited_horizon: Use minimal lookahead window (default: True, recommended)

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
        # ПРОВЕРКА: декомпозиция по генераторам создаёт фундаментальные конфликты
        # Demand/reserves constraints связывают всех генераторов. Когда партия N решается,
        # она зависит от релаксированных (дробных) значений будущих партий. Когда партия N+1
        # решается как бинарная, она не может достичь тех же значений -> infeasibility.
        #
        # Это проблема не только с limited_horizon, но и с full horizon.
        # Поэтому декомпозиция по генераторам не поддерживается.
        if generators_per_iteration < num_generators:
            if verbose:
                print(f"  WARNING: Generator decomposition ({generators_per_iteration} per iter) "
                      f"is not supported due to demand/reserves constraint coupling.")
                print(f"  Automatically using all {num_generators} generators at once (classic R&F).")
            generators_per_iteration = num_generators

        if verbose and generators_per_iteration == num_generators:
            print(f"  Classic Relax-and-Fix: {num_generators} generators at once")

    # Режим lookahead
    if verbose:
        if use_limited_horizon:
            print(f"  Minimal lookahead mode: +1 for ramping (backward unfixing handles uptime/downtime)")
        else:
            print(f"  Full horizon mode: all future periods relaxed")

    # Найти генераторы с длинным горизонтом (min_uptime или min_downtime > window_size)
    # Эти генераторы НЕ фиксируются - их переменные остаются бинарными на всём горизонте
    long_horizon_generators = _get_long_horizon_generators(data, window_size)
    if verbose and long_horizon_generators:
        print(f"  Long-horizon generators (not fixed): {len(long_horizon_generators)}")
        if len(long_horizon_generators) <= 5:
            print(f"    {', '.join(sorted(long_horizon_generators))}")

    # Основной цикл Relax-and-Fix (единая логика для всех стратегий)
    for start in range(0, num_periods, window_step):
        end = min(start + window_size, num_periods)
        step = min(start + window_step, num_periods)

        # IMPORTANT: time_periods is 1-based [1, 2, 3, ..., 48]
        # But the loop uses 0-based indices (start, end, step)
        # We need 1-based period numbers for the model
        # Example: start=0, end=8 -> periods [1..8]
        #          start=8, end=16 -> periods [9..16]
        #          start=32, end=40 -> periods [33..40]
        # So: start_period = start + 1, end_period = end (NOT end+1)
        start_period = start + 1
        end_period = end
        step_period = step

        # Create sets of 1-based period numbers
        # range(1, 9) = [1,2,3,4,5,6,7,8] for start=0, end=8
        window_periods = set(range(start_period, min(end_period + 1, num_periods + 1)))
        fix_periods = set(range(start_period, min(step_period + 1, num_periods + 1)))

        # BACKWARD UNFIXING теперь применяется для каждой партии генераторов отдельно
        # (см. внутренний цикл ниже)

        # Последнее окно обрабатывается как обычное - без полной реоптимизации

        # Определить горизонт релаксации и будущие периоды
        if use_limited_horizon:
            # Для последующих окон: используем минимальный lookahead (+1 для ramping)
            # ВАЖНО: передаём end_period, чтобы lookahead покрывал весь window
            generator_lookahead = _calculate_generator_specific_lookahead(
                step_period, end_period, data, num_periods
            )

            # Определить максимальный lookahead для создания окна релаксации
            max_lookahead = max(generator_lookahead.values())

            # Окно релаксации начинается ПОСЛЕ окна фиксации (+1 для ramping)
            lookahead_start = step_period + 1
            lookahead_end = max_lookahead
            lookahead_periods = set(range(lookahead_start, min(lookahead_end + 1, num_periods + 1)))

            # Дальние будущие периоды (за пределами ТЕКУЩЕГО ОКНА и lookahead)
            # ВАЖНО: future_periods должны начинаться ПОСЛЕ end_period, а не после lookahead!
            # Иначе периоды внутри окна будут зафиксированы на 0
            future_start = max(end_period + 1, lookahead_end + 1)
            future_periods = set(range(future_start, num_periods + 1)) if future_start <= num_periods else set()

            if verbose:
                # Show inclusive ranges in verbose output (e.g., [1:8] means periods 1,2,3,4,5,6,7,8)
                print(f"  Time Window [{start_period}:{min(end_period, num_periods)}], "
                      f"fixing [{start_period}:{min(step_period, num_periods)}], "
                      f"lookahead [{lookahead_start}:{lookahead_end}]")

            # Освободить переменные в текущем окне и окне релаксации (объединённый вызов)
            # Это может быть необходимо, если они были зафиксированы в прошлом окне
            all_active_periods = window_periods | lookahead_periods
            _unfix_variables_in_window(model, all_active_periods, binary_vars, verbose=verbose)

            # Зафиксировать переменные дальних будущих периодов на 0 (селективно)
            # ВАЖНО: long_horizon_generators исключены - они никогда не фиксируются
            if future_periods:
                _fix_future_variables_to_zero(model, future_periods, binary_vars,
                                              generator_lookahead,
                                              always_binary_generators=long_horizon_generators,
                                              verbose=verbose)
        else:
            # Полный горизонт для всех окон
            lookahead_start = step_period + 1
            lookahead_end = num_periods
            lookahead_periods = set(range(lookahead_start, min(lookahead_end + 1, num_periods + 1)))
            generator_lookahead = {g: num_periods for g in data.get("thermal_generators", {}).keys()}

            if verbose:
                # Show inclusive ranges in verbose output
                print(f"  Time Window [{start_period}:{min(end_period, num_periods)}], "
                      f"fixing [{start_period}:{min(step_period, num_periods)}]")

            # Освободить переменные в текущем окне и окне релаксации
            all_active_periods = window_periods | lookahead_periods
            _unfix_variables_in_window(model, all_active_periods, binary_vars, verbose=verbose)

        # Внутренний цикл по партиям генераторов
        for gen_start_idx in range(0, num_generators, generators_per_iteration):
            gen_end_idx = min(gen_start_idx + generators_per_iteration, num_generators)
            current_gen_batch = set(generators_sorted[gen_start_idx:gen_end_idx])

            # Вывод информации о партии (если декомпозиция включена)
            if verbose and generators_per_iteration < num_generators:
                print(f"    Generator batch [{gen_start_idx}:{gen_end_idx}] ({len(current_gen_batch)} gens)")

            # BACKWARD UNFIXING: применяется для КАЖДОЙ ПАРТИИ генераторов отдельно
            # Это критически важно - иначе разфиксированные переменные генераторов из
            # предыдущих партий создают конфликт с их уже зафиксированными переменными
            # в текущем окне
            unfixed_periods = set()
            unfixed_generators = set()
            if start > 0:
                unfixed_periods, unfixed_generators = _backward_unfix_for_startup(
                    model, start_period, data, verbose=verbose,
                    allowed_generators=current_gen_batch
                )

            # 1. Установить домены переменных
            # В режиме limited horizon: бинарные в window_periods, релаксированные в lookahead_periods
            # Также учитываем backward-unfixed переменные и long_horizon генераторы
            _set_variable_domains(binary_vars, current_gen_batch, window_periods,
                                  backward_unfixed_generators=unfixed_generators,
                                  backward_unfixed_periods=unfixed_periods,
                                  always_binary_generators=long_horizon_generators,
                                  verbose=verbose)

            # 2. Управлять ограничениями для текущей партии генераторов
            _manage_constraints_for_window(model, generator_lookahead, num_periods,
                                          active_generators=current_gen_batch,
                                          window_start=start_period,
                                          backward_unfixed_periods=unfixed_periods,
                                          backward_unfixed_generators=unfixed_generators,
                                          always_active_generators=long_horizon_generators,
                                          verbose=verbose)

            # 3. Решить подзадачу
            result, solve_time, is_optimal = _solve_subproblem(model, solver_name, gap)

            # 4. Проверить результат и вывести информацию
            termination = result.solver.termination_condition

            if termination == TerminationCondition.infeasible or \
               termination == TerminationCondition.infeasibleOrUnbounded:
                # Subproblem is infeasible - run diagnostics
                iteration_info = {
                    'start': start_period,
                    'end': end_period,
                    'step': step_period,
                    'gen_start': gen_start_idx,
                    'gen_end': gen_end_idx
                }

                # Запуск диагностики (экспорт модели и анализ)
                if data is not None:
                    diagnose_infeasibility(model, data, iteration_info, export_model=True)

                # Формирование сообщения об ошибке
                error_msg = [
                    f"\n{'='*60}",
                    "INFEASIBLE SUBPROBLEM DETECTED",
                    f"{'='*60}",
                    f"Time window: [{start_period}:{min(end_period, num_periods)}], "
                    f"fixing [{start_period}:{min(step_period, num_periods)}]",
                ]
                if use_limited_horizon and 'lookahead_end' in locals():
                    error_msg.append(f"Lookahead: [{min(step_period, num_periods)}:{lookahead_end}]")

                error_msg.extend([
                    f"Generator batch: [{gen_start_idx}:{gen_end_idx}]",
                    f"Termination condition: {termination}",
                    "",
                    "Diagnostics have been run - check output above for details.",
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
                    "- Review the exported LP file in debug_models/ directory",
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

            # 5. Зафиксировать переменные текущей партии (включая cg, pg, rg, lg!)
            # ВАЖНО: исключаем long_horizon_generators - они не фиксируются
            fixable_generators = current_gen_batch - long_horizon_generators
            _fix_variables(binary_vars, fixable_generators, fix_periods, model=model)

            # 6. Зафиксировать обратно разфиксированные периоды текущей партии из backward unfixing
            # ВАЖНО: это делается для КАЖДОЙ партии, а не в конце окна
            if unfixed_periods:
                _refix_periods(model, unfixed_periods, verbose=verbose)

        # Выход после последнего окна
        if step == num_periods:
            break

    solve_time = time.time() - start_time

    # Final LP re-optimization: зафиксировать бинарные решения, пересчитать dispatch
    # Это гарантирует feasibility continuous переменных при данных бинарных решениях
    if data is not None and model_builder is not None:
        lp_result = _final_lp_reoptimization(
            model, data, model_builder, solver_name, gap, verbose
        )
        if lp_result['success']:
            # Используем fresh модель с корректным dispatch
            final_model = lp_result['model']
            final_objective = lp_result['objective']
        else:
            final_model = model
            final_objective = value(model.obj)
            if verbose:
                print("  WARNING: LP re-optimization failed, using R&F solution as-is")
    else:
        final_model = model
        final_objective = value(model.obj)

    # Prepare result
    result = {
        'solve_time': time.time() - start_time,
        'objective': final_objective,
        'status': 'completed',
    }

    if verify_solution:
        if data is None or model_builder is None:
            raise ValueError(
                "verify_solution=True requires 'data' and 'model_builder' arguments. "
                "Pass the original problem data and model building function."
            )

        verification = _verify_solution_feasibility(
            old_model=final_model,
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
