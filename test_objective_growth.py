"""
Демонстрация роста целевой функции в методе Relax-and-Fix
"""
import json
import sys
import time
from pathlib import Path

from pyomo.environ import Binary, UnitInterval, value
from pyomo.opt import SolverFactory, TerminationCondition

sys.path.insert(0, str(Path(__file__).parent))

from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import _find_binary_variables


def relax_and_fix_with_logging(model, window_size, window_step, solver_name="appsi_highs", gap=0.01):
    """
    Relax-and-Fix с детальным логированием изменения целевой функции
    """
    # Get time periods
    time_periods = sorted(list(list(model.ug.index_set().subsets())[1]))
    num_periods = len(time_periods)

    # Find binary variables
    binary_vars = _find_binary_variables(model)

    print(f"\n{'='*80}")
    print(f"RELAX-AND-FIX: Анализ роста целевой функции")
    print(f"{'='*80}")
    print(f"Периоды: {num_periods}, Окно: {window_size}, Шаг: {window_step}")
    print(f"Бинарных переменных: {len(binary_vars)} типов")
    print(f"{'='*80}\n")

    objectives = []

    # Relax-and-Fix iterations
    for iteration, start in enumerate(range(0, num_periods, window_step), 1):
        end = min(start + window_size, num_periods)
        step = min(start + window_step, num_periods)

        if end == num_periods:
            step = end

        print(f"Итерация {iteration}: Окно [{start}:{end}], фиксация [{start}:{step}]")

        window_periods = set(time_periods[start:end])
        fix_periods = set(time_periods[start:step])

        # Count relaxed vs binary variables
        total_vars = 0
        binary_count = 0
        relaxed_count = 0
        fixed_count = 0

        # Set domains
        for var, time_idx_pos in binary_vars:
            for idx in var:
                time_period = idx[time_idx_pos]
                total_vars += 1

                if var[idx].is_fixed():
                    fixed_count += 1
                elif time_period in window_periods:
                    var[idx].domain = Binary
                    binary_count += 1
                else:
                    var[idx].domain = UnitInterval
                    relaxed_count += 1

        print(f"  Переменные: {fixed_count} зафиксировано, {binary_count} бинарных, {relaxed_count} релаксировано")

        # Solve
        iter_start = time.time()
        solver = SolverFactory(solver_name)

        if hasattr(solver, 'config'):
            solver.config.mip_gap = gap
            result = solver.solve(model, tee=False)
        else:
            result = solver.solve(model, options={'ratioGap': gap}, tee=False)

        iter_time = time.time() - iter_start

        # Get objective
        if result.solver.termination_condition == TerminationCondition.optimal:
            obj = value(model.obj)
            objectives.append(obj)

            # Calculate change from previous iteration
            if iteration > 1:
                change = obj - objectives[-2]
                change_pct = (change / objectives[-2]) * 100
                direction = "+" if change > 0 else ("-" if change < 0 else "=")
                print(f"  Целевая функция: {obj:,.2f} ({direction} {abs(change):,.2f}, {direction} {abs(change_pct):.3f}%)")
            else:
                print(f"  Целевая функция: {obj:,.2f} (начальная нижняя оценка)")

            print(f"  Время решения: {iter_time:.2f}s")
        else:
            print(f"  ОШИБКА: {result.solver.termination_condition}")
            break

        # Fix variables leaving the window
        for var, time_idx_pos in binary_vars:
            for idx in var:
                if idx[time_idx_pos] in fix_periods and not var[idx].is_fixed():
                    var[idx].fix()

        print()

        if step == num_periods:
            break

    # Summary
    print(f"{'='*80}")
    print(f"ИТОГОВЫЙ АНАЛИЗ")
    print(f"{'='*80}\n")

    print("Эволюция целевой функции:")
    for i, obj in enumerate(objectives, 1):
        if i == 1:
            print(f"  Итерация {i}: {obj:,.2f} (максимальная релаксация -> лучшая нижняя оценка)")
        elif i == len(objectives):
            print(f"  Итерация {i}: {obj:,.2f} (полная бинаризация -> истинное решение)")
        else:
            change = obj - objectives[i-2]
            print(f"  Итерация {i}: {obj:,.2f} (+ {change:,.2f})")

    total_increase = objectives[-1] - objectives[0]
    total_increase_pct = (total_increase / objectives[0]) * 100

    print(f"\nОбщий рост целевой функции: {total_increase:,.2f} ({total_increase_pct:.2f}%)")
    print(f"Начальная нижняя оценка: {objectives[0]:,.2f}")
    print(f"Финальное решение: {objectives[-1]:,.2f}")

    print(f"\n{'='*80}")
    print("ОБЪЯСНЕНИЕ:")
    print(f"{'='*80}")
    print("""
На каждой итерации мы добавляем ограничения бинарности, что СУЖАЕТ допустимое
множество решений. В задаче МИНИМИЗАЦИИ это означает, что оптимум может только
УХУДШИТЬСЯ (увеличиться) или остаться прежним.

Математически:
  - Релаксированная задача: x ∈ [0,1]ⁿ → большое допустимое множество
  - Бинарная задача: x ∈ {0,1}ⁿ → меньшее допустимое множество
  - Поскольку {0,1}ⁿ ⊂ [0,1]ⁿ, оптимум может только ухудшиться!

Это НОРМАЛЬНОЕ и ОЖИДАЕМОЕ поведение Relax-and-Fix!
    """)


def main():
    # Используем меньший пример для быстрой демонстрации
    DATA_FILE = r"C:\Users\oQaris\Desktop\Git\uc\examples\rts_gmlc\2020-07-06.json"

    print(f"Загрузка данных из {DATA_FILE}")
    with open(DATA_FILE) as f:
        data = json.load(f)

    print(f"Задача: {data['time_periods']} периодов, {len(data['thermal_generators'])} генераторов\n")

    # Build model
    print("Построение модели...")
    model = build_uc_model(data)

    # Run Relax-and-Fix with logging
    relax_and_fix_with_logging(model, window_size=8, window_step=4, solver_name="appsi_highs", gap=0.0)


if __name__ == "__main__":
    main()
