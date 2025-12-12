"""
Сравнение различных стратегий расчета lookahead окна
"""
import json
import sys

sys.path.insert(0, 'src')

from models.uc_model import build_uc_model
from solvers.relax_and_fix import solve_relax_and_fix


def test_lookahead_strategies():
    """
    Сравнение разных стратегий: median, percentile75, percentile90, max
    """
    print("=" * 80)
    print("Comparison of Lookahead Window Calculation Strategies")
    print("=" * 80)

    # Загрузить тестовый пример
    test_file = r"examples\rts_gmlc\2020-07-06.json"
    print(f"\nTest instance: {test_file}")

    with open(test_file, 'r') as f:
        data = json.load(f)

    print(f"  Time periods: {data['time_periods']}")
    print(f"  Thermal generators: {len(data['thermal_generators'])}")

    # Параметры алгоритма
    window_size = 6
    window_step = 4
    gap = 0.01
    solver_name = "appsi_highs"

    # Стратегии для тестирования
    strategies = ['median', 'percentile75', 'percentile90']

    results = {}

    for strategy in strategies:
        print("\n" + "=" * 80)
        print(f"Strategy: {strategy.upper()}")
        print("=" * 80)

        model = build_uc_model(data)

        result = solve_relax_and_fix(
            model=model,
            window_size=window_size,
            window_step=window_step,
            gap=gap,
            solver_name=solver_name,
            verbose=True,
            verify_solution=True,
            data=data,
            model_builder=build_uc_model,
            use_limited_horizon=True,
            lookahead_strategy=strategy
        )

        results[strategy] = result

        print(f"\n  Strategy: {strategy}")
        print(f"  Solve time: {result['solve_time']:.2f}s")
        print(f"  Objective: ${result['objective']:.2f}")
        print(f"  Feasible: {result.get('feasible', 'N/A')}")
        if result.get('feasible'):
            print(f"  Verified objective: ${result['verified_objective']:.2f}")

    # Сравнительная таблица
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<15} {'Time (s)':<12} {'Objective':<15} {'Feasible':<10} {'Speedup':<10}")
    print("-" * 80)

    # Baseline для сравнения - percentile75
    baseline_time = results['percentile75']['solve_time']

    for strategy in strategies:
        r = results[strategy]
        speedup = baseline_time / r['solve_time'] if r['solve_time'] > 0 else 0
        obj = r.get('verified_objective', r['objective'])
        feasible = 'YES' if r.get('feasible') else 'NO'

        print(f"{strategy:<15} {r['solve_time']:<12.2f} ${obj:<14.2f} {feasible:<10} {speedup:<10.2f}x")

    # Качество решений
    if all(r.get('feasible') for r in results.values()):
        print("\n" + "=" * 80)
        print("OBJECTIVE QUALITY (vs percentile75)")
        print("=" * 80)

        baseline_obj = results['percentile75']['verified_objective']

        for strategy in strategies:
            obj = results[strategy]['verified_objective']
            diff = obj - baseline_obj
            diff_pct = (diff / baseline_obj) * 100 if baseline_obj > 0 else 0

            print(f"{strategy:<15} ${obj:>12.2f}  (diff: ${diff:>10.2f}, {diff_pct:>6.2f}%)")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Найти лучшую стратегию по времени
    fastest = min(strategies, key=lambda s: results[s]['solve_time'])
    print(f"Fastest: {fastest} ({results[fastest]['solve_time']:.2f}s)")

    # Найти лучшую по качеству (если все допустимы)
    if all(r.get('feasible') for r in results.values()):
        best_obj = min(strategies, key=lambda s: results[s]['verified_objective'])
        print(f"Best objective: {best_obj} (${results[best_obj]['verified_objective']:.2f})")

    print("\nRecommendation: Use 'percentile75' for good balance between speed and quality.")


if __name__ == "__main__":
    test_lookahead_strategies()
