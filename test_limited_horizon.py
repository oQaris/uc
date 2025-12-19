"""
Тестовый скрипт для проверки алгоритма Relax-and-Fix с ограниченным горизонтом
"""
import json
import sys
import time

sys.path.insert(0, 'src')

from models.uc_model import build_uc_model
from solvers.relax_and_fix import solve_relax_and_fix


def test_limited_horizon():
    """
    Тестирование алгоритма с ограниченным горизонтом на небольшом примере
    """
    print("=" * 80)
    print("Testing Relax-and-Fix with Limited Horizon")
    print("=" * 80)

    # Загрузить небольшой тестовый пример (RTS-GMLC)
    test_file = r"examples\rts_gmlc\2020-07-06.json"
    print(f"\nLoading test instance: {test_file}")

    with open(test_file, 'r') as f:
        data = json.load(f)

    print(f"  Time periods: {data['time_periods']}")
    print(f"  Thermal generators: {len(data['thermal_generators'])}")
    print(f"  Renewable generators: {len(data['renewable_generators'])}")

    # Построить модель
    print("\nBuilding UC model...")
    model = build_uc_model(data)

    # Параметры алгоритма
    window_size = 6
    window_step = 4
    gap = 0.01
    solver_name = "appsi_highs"

    print("\n" + "=" * 80)
    print("Test 1: Limited Horizon Mode (NEW)")
    print("=" * 80)

    # Тест 1: С ограниченным горизонтом (новый режим)
    model_limited = build_uc_model(data)
    start_time = time.time()

    result_limited = solve_relax_and_fix(
        model=model_limited,
        window_size=window_size,
        window_step=window_step,
        gap=gap,
        solver_name=solver_name,
        verbose=True,
        verify_solution=True,
        data=data,
        model_builder=build_uc_model,
        use_limited_horizon=True,  # Новый режим!
        lookahead_strategy='percentile75'  # Рекомендуемая стратегия
    )

    elapsed_limited = time.time() - start_time

    print("\n" + "-" * 80)
    print("Results (Limited Horizon):")
    print(f"  Total solve time: {result_limited['solve_time']:.2f}s")
    print(f"  Objective: ${result_limited['objective']:.2f}")
    if 'feasible' in result_limited:
        print(f"  Verified feasibility: {result_limited['feasible']}")
        if result_limited['feasible']:
            print(f"  Verified objective: ${result_limited['verified_objective']:.2f}")
            print(f"  Objective gap: ${result_limited['objective_gap']:.2f}")

    print("\n" + "=" * 80)
    print("Test 2: Full Horizon Mode (CLASSIC)")
    print("=" * 80)

    # Тест 2: Полный горизонт (классический режим) для сравнения
    model_full = build_uc_model(data)
    start_time = time.time()

    result_full = solve_relax_and_fix(
        model=model_full,
        window_size=window_size,
        window_step=window_step,
        gap=gap,
        solver_name=solver_name,
        verbose=True,
        verify_solution=True,
        data=data,
        model_builder=build_uc_model,
        use_limited_horizon=False  # Классический режим
    )

    elapsed_full = time.time() - start_time

    print("\n" + "-" * 80)
    print("Results (Full Horizon):")
    print(f"  Total solve time: {result_full['solve_time']:.2f}s")
    print(f"  Objective: ${result_full['objective']:.2f}")
    if 'feasible' in result_full:
        print(f"  Verified feasibility: {result_full['feasible']}")
        if result_full['feasible']:
            print(f"  Verified objective: ${result_full['verified_objective']:.2f}")
            print(f"  Objective gap: ${result_full['objective_gap']:.2f}")

    # Сравнение результатов
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"Limited Horizon:")
    print(f"  Solve time: {result_limited['solve_time']:.2f}s")
    print(f"  Objective: ${result_limited.get('verified_objective', result_limited['objective']):.2f}")
    print(f"  Feasible: {result_limited.get('feasible', 'N/A')}")

    print(f"\nFull Horizon:")
    print(f"  Solve time: {result_full['solve_time']:.2f}s")
    print(f"  Objective: ${result_full.get('verified_objective', result_full['objective']):.2f}")
    print(f"  Feasible: {result_full.get('feasible', 'N/A')}")

    # Выводы
    speedup = result_full['solve_time'] / result_limited['solve_time'] if result_limited['solve_time'] > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")

    if result_limited.get('feasible') and result_full.get('feasible'):
        obj_diff = abs(result_limited['verified_objective'] - result_full['verified_objective'])
        obj_pct = (obj_diff / result_full['verified_objective']) * 100
        print(f"Objective difference: ${obj_diff:.2f} ({obj_pct:.2f}%)")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_limited_horizon()
