"""
Диагностические утилиты для отладки Relax-and-Fix
"""
import os

from pyomo.environ import value, Constraint, Binary
from pyomo.core.plugins.transform.relax_integrality import RelaxIntegrality


def export_model_state(model, iteration_info, output_dir="debug_models"):
    """
    Экспортировать состояние модели в файл для анализа

    Args:
        model: Pyomo ConcreteModel
        iteration_info: dict с информацией о текущей итерации
        output_dir: директория для сохранения файлов

    Returns:
        str: путь к сохраненному файлу
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # RelaxIntegrality().apply_to(model)

    # Формируем имя файла
    time_window = f"t{iteration_info['start']}-{iteration_info['end']}"
    gen_batch = f"g{iteration_info['gen_start']}-{iteration_info['gen_end']}"
    filename = f"model_{time_window}_{gen_batch}.lp"
    filepath = os.path.join(output_dir, filename)

    # Экспортируем модель в LP формат
    model.write(filepath, io_options={'symbolic_solver_labels': True})

    print(f"    Model exported to: {filepath}")
    return filepath


def analyze_fixed_variables(model, iteration_info):
    """
    Анализировать зафиксированные переменные

    Args:
        model: Pyomo ConcreteModel
        iteration_info: dict с информацией о текущей итерации
    """
    from pyomo.environ import Var

    print(f"\n  {'=' * 60}")
    print(f"  FIXED VARIABLES ANALYSIS")
    print(f"  {'=' * 60}")

    # Подсчет зафиксированных переменных по типам
    fixed_vars = {}

    for component in model.component_objects(ctype=Var):
        var_name = component.name
        fixed_count = 0
        fixed_values = {}

        for idx in component:
            if component[idx].is_fixed():
                fixed_count += 1
                val = value(component[idx])
                if val not in fixed_values:
                    fixed_values[val] = 0
                fixed_values[val] += 1

        if fixed_count > 0:
            fixed_vars[var_name] = {
                'count': fixed_count,
                'values': fixed_values,
                'total': len(component)
            }

    # Вывод информации
    for var_name, info in sorted(fixed_vars.items()):
        percentage = (info['count'] / info['total']) * 100
        print(f"  {var_name}: {info['count']}/{info['total']} fixed ({percentage:.1f}%)")
        for val, count in sorted(info['values'].items()):
            print(f"    value={val}: {count} vars")

    print(f"  {'=' * 60}\n")


def analyze_active_constraints(model):
    """
    Анализировать активные и неактивные ограничения

    Args:
        model: Pyomo ConcreteModel
    """
    print(f"\n  {'=' * 60}")
    print(f"  CONSTRAINTS ANALYSIS")
    print(f"  {'=' * 60}")

    constraint_stats = {}

    for component in model.component_objects(ctype=Constraint):
        const_name = component.name
        active_count = 0
        inactive_count = 0

        for idx in component:
            if component[idx].active:
                active_count += 1
            else:
                inactive_count += 1

        total = active_count + inactive_count
        constraint_stats[const_name] = {
            'active': active_count,
            'inactive': inactive_count,
            'total': total
        }

    # Вывод информации
    for const_name, stats in sorted(constraint_stats.items()):
        if stats['total'] > 0:
            active_pct = (stats['active'] / stats['total']) * 100
            print(f"  {const_name}: {stats['active']}/{stats['total']} active ({active_pct:.1f}%)")

    print(f"  {'=' * 60}\n")


def check_demand_feasibility(model, data, iteration_info):
    """
    Проверить, может ли спрос быть удовлетворен текущими переменными

    Args:
        model: Pyomo ConcreteModel
        data: исходные данные
        iteration_info: dict с информацией о текущей итерации
    """
    print(f"\n  {'=' * 60}")
    print(f"  DEMAND FEASIBILITY CHECK")
    print(f"  {'=' * 60}")

    thermal_gens = data.get("thermal_generators", {})
    demand = data.get("demand", [])

    # Для каждого периода в текущем окне проверяем доступную мощность
    start = iteration_info['start']
    end = iteration_info['end']

    for t in range(start, end + 1):  # Включаем end период
        # ВАЖНО: t - это 1-based индекс из модели (1, 2, 3, ...),
        # а demand - это 0-based список, поэтому используем t-1
        t_idx = t - 1
        if t_idx < 0 or t_idx >= len(demand):
            continue

        period_demand = demand[t_idx]

        # Подсчитываем доступную мощность с учетом зафиксированных переменных
        available_power = 0
        fixed_off_power = 0  # Мощность генераторов, зафиксированных в OFF

        for gen_name, gen_data in thermal_gens.items():
            max_power = gen_data.get("power_output_maximum", 0)

            if (gen_name, t) in model.ug:
                ug_var = model.ug[gen_name, t]

                if ug_var.is_fixed():
                    if value(ug_var) == 1:
                        available_power += max_power
                    else:
                        fixed_off_power += max_power
                else:
                    # Переменная не зафиксирована - может быть включена
                    available_power += max_power

        deficit = period_demand - available_power

        if deficit > 0:
            print(f"  Period {t}: POTENTIAL DEFICIT")
            print(f"    Demand: {period_demand:.2f} MW")
            print(f"    Available: {available_power:.2f} MW")
            print(f"    Deficit: {deficit:.2f} MW")
            print(f"    Fixed OFF power: {fixed_off_power:.2f} MW")
        else:
            surplus = available_power - period_demand
            print(f"  Period {t}: OK (surplus: {surplus:.2f} MW)")

    print(f"  {'=' * 60}\n")


def analyze_variable_domains(model, iteration_info):
    """
    Анализировать домены переменных в текущем окне

    Args:
        model: Pyomo ConcreteModel
        iteration_info: dict с информацией о текущей итерации
    """
    from pyomo.environ import Var, UnitInterval

    print(f"\n  {'=' * 60}")
    print(f"  VARIABLE DOMAINS ANALYSIS")
    print(f"  {'=' * 60}")

    start = iteration_info['start']
    end = iteration_info['end']

    domain_stats = {}

    for component in model.component_objects(ctype=Var):
        var_name = component.name
        binary_count = 0
        relaxed_count = 0
        fixed_count = 0

        for idx in component:
            # Check if variable is in current window
            # For (g, t) variables, time is at position 1
            # For (g, s, t) variables, time is at position 2
            if len(idx) == 2:
                time_period = idx[1]
            elif len(idx) == 3:
                time_period = idx[2]
            else:
                continue

            if start <= time_period <= end:  # Включаем end период
                if component[idx].is_fixed():
                    fixed_count += 1
                elif component[idx].domain == Binary:
                    binary_count += 1
                elif component[idx].domain == UnitInterval:
                    relaxed_count += 1

        if binary_count > 0 or relaxed_count > 0:
            domain_stats[var_name] = {
                'binary': binary_count,
                'relaxed': relaxed_count,
                'fixed': fixed_count
            }

    # Вывод информации
    for var_name, stats in sorted(domain_stats.items()):
        total = stats['binary'] + stats['relaxed'] + stats['fixed']
        print(f"  {var_name}: {stats['binary']} Binary, {stats['relaxed']} Relaxed, {stats['fixed']} Fixed (total: {total})")

    print(f"  {'=' * 60}\n")


def diagnose_infeasibility(model, data, iteration_info, export_model=True):
    """
    Комплексная диагностика недостижимости

    Args:
        model: Pyomo ConcreteModel (перед решением недопустимой подзадачи)
        data: исходные данные
        iteration_info: dict с ключами 'start', 'end', 'gen_start', 'gen_end'
        export_model: экспортировать модель в LP файл

    Returns:
        dict: информация о диагностике
    """
    print(f"\n{'#' * 70}")
    print(f"# INFEASIBILITY DIAGNOSTICS")
    print(f"# Time window: [{iteration_info['start']}:{iteration_info['end']}]")
    print(f"# Generator batch: [{iteration_info['gen_start']}:{iteration_info['gen_end']}]")
    print(f"{'#' * 70}")

    # 1. Экспорт модели
    model_file = None
    if export_model:
        model_file = export_model_state(model, iteration_info)

    # 2. Анализ зафиксированных переменных
    analyze_fixed_variables(model, iteration_info)

    # 3. Анализ доменов переменных
    analyze_variable_domains(model, iteration_info)

    # 4. Анализ активных ограничений
    analyze_active_constraints(model)

    # 5. Проверка выполнимости спроса
    check_demand_feasibility(model, data, iteration_info)

    print(f"\n{'#' * 70}")
    print(f"# SUGGESTIONS:")
    print(f"# 1. Check the exported LP file: {model_file}")
    print(f"# 2. Review fixed variables - are they creating conflicts?")
    print(f"# 3. Check demand feasibility - is there enough available power?")
    print(f"# 4. Review inactive constraints - are critical constraints disabled?")
    print(f"{'#' * 70}\n")

    return {
        'model_file': model_file,
        'iteration_info': iteration_info
    }


def check_constraint_violations(model, tolerance=1e-6):
    """
    Проверить нарушения ограничений в текущем решении

    Args:
        model: Pyomo ConcreteModel с решением
        tolerance: допуск для нарушений

    Returns:
        list: список нарушенных ограничений
    """
    violations = []

    for component in model.component_objects(ctype=Constraint):
        for idx in component:
            constraint = component[idx]

            if not constraint.active:
                continue

            try:
                # Вычисляем значение тела ограничения
                body_value = value(constraint.body)

                # Проверяем нижнюю границу
                if constraint.lower is not None:
                    lower_value = value(constraint.lower)
                    if body_value < lower_value - tolerance:
                        violations.append({
                            'constraint': f"{component.name}[{idx}]",
                            'type': 'lower_bound',
                            'body': body_value,
                            'bound': lower_value,
                            'violation': lower_value - body_value
                        })

                # Проверяем верхнюю границу
                if constraint.upper is not None:
                    upper_value = value(constraint.upper)
                    if body_value > upper_value + tolerance:
                        violations.append({
                            'constraint': f"{component.name}[{idx}]",
                            'type': 'upper_bound',
                            'body': body_value,
                            'bound': upper_value,
                            'violation': body_value - upper_value
                        })
            except:
                # Не можем вычислить значение - пропускаем
                pass

    return violations


def print_constraint_violations(violations, max_show=20):
    """
    Вывести информацию о нарушениях ограничений

    Args:
        violations: список нарушений от check_constraint_violations
        max_show: максимальное количество нарушений для вывода
    """
    if not violations:
        print("  No constraint violations found!")
        return

    print(f"\n  {'=' * 60}")
    print(f"  CONSTRAINT VIOLATIONS: {len(violations)} found")
    print(f"  {'=' * 60}")

    # Сортируем по величине нарушения
    violations.sort(key=lambda x: abs(x['violation']), reverse=True)

    for i, v in enumerate(violations[:max_show]):
        print(f"  {i + 1}. {v['constraint']}")
        print(f"     Type: {v['type']}")
        print(f"     Body: {v['body']:.6f}, Bound: {v['bound']:.6f}")
        print(f"     Violation: {v['violation']:.6e}")

    if len(violations) > max_show:
        print(f"  ... and {len(violations) - max_show} more violations")

    print(f"  {'=' * 60}\n")
