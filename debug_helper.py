"""
Вспомогательный скрипт для ручной диагностики при отладке

Использование в отладчике (breakpoint):
1. Поставьте точку останова в relax_and_fix.py перед вызовом _solve_subproblem
2. В консоли отладчика импортируйте этот модуль:
   >>> from debug_helper import debug_current_state
3. Вызовите функцию с текущими переменными:
   >>> debug_current_state(model, data, start, end, gen_start_idx, gen_end_idx)
"""

from src.solvers.diagnostics import (
    export_model_state,
    analyze_fixed_variables,
    analyze_active_constraints,
    check_demand_feasibility,
    diagnose_infeasibility
)


def debug_current_state(model, data, start, end, gen_start_idx, gen_end_idx, step=None):
    """
    Отладочная функция для вызова из дебаггера

    Args:
        model: текущая Pyomo модель
        data: исходные данные задачи
        start: начало временного окна
        end: конец временного окна
        gen_start_idx: начало партии генераторов
        gen_end_idx: конец партии генераторов
        step: конец окна фиксации (опционально)
    """
    iteration_info = {
        'start': start,
        'end': end,
        'step': step if step is not None else end,
        'gen_start': gen_start_idx,
        'gen_end': gen_end_idx
    }

    print("\n" + "="*70)
    print("MANUAL DEBUGGING SESSION")
    print("="*70 + "\n")

    # Запускаем полную диагностику
    result = diagnose_infeasibility(model, data, iteration_info, export_model=True)

    print("\nDEBUGGING SESSION COMPLETE")
    print("="*70 + "\n")

    return result


def quick_export(model, filename="debug_model.lp"):
    """
    Быстрый экспорт модели в файл

    Args:
        model: Pyomo модель
        filename: имя файла для сохранения
    """
    model.write(filename, io_options={'symbolic_solver_labels': True})
    print(f"Model exported to: {filename}")
    return filename


def check_variable(model, var_name, index=None):
    """
    Проверить состояние переменной

    Args:
        model: Pyomo модель
        var_name: имя переменной (например, 'ug')
        index: индекс переменной (опционально, если None - показать все)

    Examples:
        >>> check_variable(model, 'ug', ('gen1', 5))  # проверить ug['gen1', 5]
        >>> check_variable(model, 'ug')  # показать статистику по всем ug
    """
    from pyomo.environ import value

    if not hasattr(model, var_name):
        print(f"Variable '{var_name}' not found in model")
        return

    var = getattr(model, var_name)

    if index is not None:
        # Проверить конкретную переменную
        if index in var:
            v = var[index]
            print(f"\n{var_name}[{index}]:")
            print(f"  Value: {value(v)}")
            print(f"  Fixed: {v.is_fixed()}")
            print(f"  Domain: {v.domain}")
            print(f"  Bounds: lb={v.lb}, ub={v.ub}")
        else:
            print(f"Index {index} not found in variable {var_name}")
    else:
        # Показать статистику
        total = len(var)
        fixed = sum(1 for idx in var if var[idx].is_fixed())
        nonzero = sum(1 for idx in var if abs(value(var[idx])) > 1e-6)

        print(f"\n{var_name} statistics:")
        print(f"  Total variables: {total}")
        print(f"  Fixed: {fixed} ({fixed/total*100:.1f}%)")
        print(f"  Non-zero values: {nonzero} ({nonzero/total*100:.1f}%)")

        # Показать несколько примеров
        print(f"  First 5 variables:")
        for i, idx in enumerate(var):
            if i >= 5:
                break
            v = var[idx]
            status = "FIXED" if v.is_fixed() else "free"
            print(f"    {var_name}[{idx}] = {value(v):.6f} ({status})")


def check_constraint(model, const_name, index=None):
    """
    Проверить состояние ограничения

    Args:
        model: Pyomo модель
        const_name: имя ограничения (например, 'demand')
        index: индекс ограничения (опционально)

    Examples:
        >>> check_constraint(model, 'demand', 5)  # проверить demand[5]
        >>> check_constraint(model, 'uptime')  # статистика по всем uptime
    """
    from pyomo.environ import value

    if not hasattr(model, const_name):
        print(f"Constraint '{const_name}' not found in model")
        return

    const = getattr(model, const_name)

    if index is not None:
        # Проверить конкретное ограничение
        if index in const:
            c = const[index]
            print(f"\n{const_name}[{index}]:")
            print(f"  Active: {c.active}")
            try:
                body_val = value(c.body)
                print(f"  Body value: {body_val}")
                if c.lower is not None:
                    print(f"  Lower bound: {value(c.lower)}")
                    print(f"  Lower slack: {body_val - value(c.lower)}")
                if c.upper is not None:
                    print(f"  Upper bound: {value(c.upper)}")
                    print(f"  Upper slack: {value(c.upper) - body_val}")
            except:
                print("  Cannot evaluate (contains unfixed variables)")
        else:
            print(f"Index {index} not found in constraint {const_name}")
    else:
        # Показать статистику
        total = len(const)
        active = sum(1 for idx in const if const[idx].active)

        print(f"\n{const_name} statistics:")
        print(f"  Total constraints: {total}")
        print(f"  Active: {active} ({active/total*100:.1f}%)")
        print(f"  Inactive: {total-active}")


def interactive_debug():
    """
    Интерактивная отладочная сессия

    Вызывайте из точки останова:
    >>> from debug_helper import interactive_debug
    >>> interactive_debug()
    """
    print("\n" + "="*70)
    print("INTERACTIVE DEBUG MODE")
    print("="*70)
    print("\nAvailable commands:")
    print("  help() - show this help")
    print("  export(model, 'file.lp') - export model to file")
    print("  var(model, 'ug', ('gen1', 5)) - check variable")
    print("  const(model, 'demand', 5) - check constraint")
    print("  exit() - exit debug mode")
    print("\nTip: You have access to all functions from diagnostics module")
    print("="*70 + "\n")


# Создаем короткие алиасы для удобства
export = quick_export
var = check_variable
const = check_constraint


if __name__ == "__main__":
    print(__doc__)
    print("\nThis module is meant to be imported in a debugger.")
    print("Set a breakpoint in relax_and_fix.py and use these functions.")
