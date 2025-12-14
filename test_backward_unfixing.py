"""
Тест нового подхода с backward unfixing для Relax-and-Fix

Сравнение трёх стратегий:
1. НОВЫЙ: minimal lookahead + backward unfixing
2. LEGACY: percentile75 lookahead (старый подход)
3. CONSERVATIVE: conservative lookahead + backward unfixing
"""
import json
import time
from pathlib import Path

from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix


def test_backward_unfixing(data_file, verbose=True):
    """
    Тестирование backward unfixing на одном файле данных

    Args:
        data_file: путь к JSON файлу с данными
        verbose: детальное логгирование
    """
    print("=" * 80)
    print(f"Testing: {data_file.name}")
    print("=" * 80)

    # Загрузить данные
    with open(data_file, 'r') as f:
        data = json.load(f)

    num_gens = len(data.get('thermal_generators', {}))
    time_periods = data.get('time_periods', 24)

    print(f"\nInstance info:")
    print(f"  Thermal generators: {num_gens}")
    print(f"  Time periods: {time_periods}")

    # Получить статистику по UTg/DTg
    uptime_list = []
    downtime_list = []
    for g, gen_data in data.get('thermal_generators', {}).items():
        uptime_list.append(gen_data.get('time_up_minimum', 0))
        downtime_list.append(gen_data.get('time_down_minimum', 0))

    max_uptime = max(uptime_list) if uptime_list else 0
    max_downtime = max(downtime_list) if downtime_list else 0

    print(f"  Max uptime:   {max_uptime} periods")
    print(f"  Max downtime: {max_downtime} periods")
    print()

    strategies = [
        {
            'name': 'NEW: minimal + backward unfixing',
            'lookahead_strategy': 'minimal',
            'use_backward_unfixing': True,
        },
        {
            'name': 'LEGACY: percentile75 (no backward)',
            'lookahead_strategy': 'legacy',
            'use_backward_unfixing': False,
        },
        {
            'name': 'CONSERVATIVE: conservative + backward',
            'lookahead_strategy': 'conservative',
            'use_backward_unfixing': True,
        },
    ]

    results = []

    for strategy in strategies:
        print("\n" + "-" * 80)
        print(f"Strategy: {strategy['name']}")
        print("-" * 80)

        # Построить свежую модель
        model = build_uc_model(data)

        # Запустить Relax-and-Fix
        start = time.time()
        result = solve_relax_and_fix(
            model=model,
            window_size=6,
            window_step=4,
            gap=0.01,
            solver_name='appsi_highs',
            verbose=verbose,
            verify_solution=True,
            data=data,
            model_builder=build_uc_model,
            generators_per_iteration=None,  # все генераторы сразу
            use_limited_horizon=True,
            lookahead_strategy=strategy['lookahead_strategy'],
            use_backward_unfixing=strategy['use_backward_unfixing']
        )
        elapsed = time.time() - start

        # Сохранить результаты
        result['strategy_name'] = strategy['name']
        result['total_time'] = elapsed
        results.append(result)

        print(f"\nResults:")
        print(f"  Total time: {elapsed:.2f}s")
        obj = result.get('objective')
        obj_str = f"{obj:.2f}" if obj is not None else "N/A"
        print(f"  Objective:  {obj_str}")
        print(f"  Feasible:   {result.get('feasible', 'N/A')}")
        if result.get('verification'):
            ver_obj = result['verification'].get('objective')
            ver_gap = result['verification'].get('gap')
            ver_obj_str = f"{ver_obj:.2f}" if ver_obj is not None else "N/A"
            ver_gap_str = f"{ver_gap:.2f}" if ver_gap is not None else "N/A"
            print(f"  Verified objective: {ver_obj_str}")
            print(f"  Objective gap:      {ver_gap_str}")

    # Сравнительная таблица
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<45} {'Time (s)':>10} {'Objective':>12} {'Feasible':>10}")
    print("-" * 80)

    for r in results:
        strategy_name = r['strategy_name']
        total_time = r.get('total_time', 0)
        objective = r.get('verified_objective') or r.get('objective')
        feasible = 'YES' if r.get('feasible', False) else 'NO'

        obj_str = f"{objective:>12.2f}" if objective is not None else f"{'N/A':>12}"
        print(f"{strategy_name:<45} {total_time:>10.2f} {obj_str} {feasible:>10}")

    print("=" * 80)

    # Определить победителя
    feasible_results = [r for r in results if r.get('feasible', False)]
    if feasible_results:
        best = min(feasible_results, key=lambda r: r.get('total_time', float('inf')))
        print(f"\nFASTEST feasible strategy: {best['strategy_name']}")
        print(f"  Time: {best.get('total_time', 0):.2f}s")
        print(f"  Objective: {best.get('verified_objective') or best.get('objective', 0):.2f}")
    else:
        print("\nWARNING: No feasible solutions found!")

    return results


if __name__ == '__main__':
    # Тестировать на небольшом примере из RTS-GMLC
    examples_dir = Path(__file__).parent / 'examples' / 'rts_gmlc'

    # Выбрать один файл для теста
    test_files = sorted(examples_dir.glob('*.json'))

    if not test_files:
        print("ERROR: No test files found in examples/rts_gmlc/")
        exit(1)

    # Использовать первый файл
    test_file = test_files[0]

    print(f"Testing backward unfixing approach on: {test_file.name}")
    print()

    results = test_backward_unfixing(test_file, verbose=True)

    print("\nTest completed!")
