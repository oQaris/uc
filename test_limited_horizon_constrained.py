"""
Тестовый скрипт для проверки алгоритма Relax-and-Fix с принудительным ограничением lookahead окна
"""
import json
import sys
import time

sys.path.insert(0, 'src')

from models.uc_model import build_uc_model
from solvers.relax_and_fix import solve_relax_and_fix


def test_constrained_lookahead():
    """
    Тестирование алгоритма с принудительным ограничением lookahead окна
    """
    print("=" * 80)
    print("Testing Relax-and-Fix with Constrained Lookahead Window")
    print("=" * 80)

    # Загрузить небольшой тестовый пример (RTS-GMLC)
    test_file = r"examples\rts_gmlc\2020-07-06.json"
    print(f"\nLoading test instance: {test_file}")

    with open(test_file, 'r') as f:
        data = json.load(f)

    print(f"  Time periods: {data['time_periods']}")
    print(f"  Thermal generators: {len(data['thermal_generators'])}")
    print(f"  Renewable generators: {len(data['renewable_generators'])}")

    # Параметры алгоритма
    window_size = 6
    window_step = 4
    gap = 0.01
    solver_name = "appsi_highs"

    print("\n" + "=" * 80)
    print("Test 1: Limited Horizon with max_lookahead=12 (FORCED)")
    print("=" * 80)

    # Тест 1: С принудительным ограничением lookahead=12
    model_constrained = build_uc_model(data)
    start_time = time.time()

    result_constrained = solve_relax_and_fix(
        model=model_constrained,
        window_size=window_size,
        window_step=window_step,
        gap=gap,
        solver_name=solver_name,
        verbose=True,
        verify_solution=True,
        data=data,
        model_builder=build_uc_model,
        use_limited_horizon=True,
        max_lookahead_periods=12  # Принудительно ограничить lookahead
    )

    elapsed_constrained = time.time() - start_time

    print("\n" + "-" * 80)
    print("Results (Constrained Lookahead):")
    print(f"  Total solve time: {result_constrained['solve_time']:.2f}s")
    print(f"  Objective: ${result_constrained['objective']:.2f}")
    if 'feasible' in result_constrained:
        print(f"  Verified feasibility: {result_constrained['feasible']}")
        if result_constrained['feasible']:
            print(f"  Verified objective: ${result_constrained['verified_objective']:.2f}")
            print(f"  Objective gap: ${result_constrained['objective_gap']:.2f}")

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

    print(f"Constrained Lookahead (max=12):")
    print(f"  Solve time: {result_constrained['solve_time']:.2f}s")
    print(f"  Objective: ${result_constrained.get('verified_objective', result_constrained['objective']):.2f}")
    print(f"  Feasible: {result_constrained.get('feasible', 'N/A')}")

    print(f"\nFull Horizon:")
    print(f"  Solve time: {result_full['solve_time']:.2f}s")
    print(f"  Objective: ${result_full.get('verified_objective', result_full['objective']):.2f}")
    print(f"  Feasible: {result_full.get('feasible', 'N/A')}")

    # Выводы
    speedup = result_full['solve_time'] / result_constrained['solve_time'] if result_constrained['solve_time'] > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")

    if result_constrained.get('feasible') and result_full.get('feasible'):
        obj_diff = abs(result_constrained['verified_objective'] - result_full['verified_objective'])
        obj_pct = (obj_diff / result_full['verified_objective']) * 100
        print(f"Objective difference: ${obj_diff:.2f} ({obj_pct:.2f}%)")
    elif not result_constrained.get('feasible'):
        print("\n⚠️ WARNING: Constrained lookahead produced INFEASIBLE solution!")
        print("   This may indicate that max_lookahead_periods is too small")
        print("   for the problem's min up/down time constraints.")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_constrained_lookahead()
