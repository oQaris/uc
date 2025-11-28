"""
Quick test for windowed Relax-and-Fix implementation
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix


def main():
    # Load small test instance
    data_file = Path("examples/rts_gmlc/2020-07-06.json")

    print(f"Loading data from {data_file}")
    with open(data_file) as f:
        data = json.load(f)

    print(f"Instance: {data['time_periods']} periods, {len(data['thermal_generators'])} thermal gens")

    # Test with small window
    print("\n" + "=" * 60)
    print("TESTING WINDOWED RELAX-AND-FIX")
    print("=" * 60)

    model = build_uc_model(data)

    result = solve_relax_and_fix(
        model,
        window_size=8,
        window_step=4,
        solver_name="appsi_highs",
        gap=0.01,
        verbose=True,
        data=data,
        model_builder=build_uc_model,
        verify_solution=True
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Solve time: {result['solve_time']:.2f}s")
    print(f"Objective: {result['objective']:.2f}")

    if 'feasible' in result:
        feasible_str = "FEASIBLE" if result['feasible'] else "INFEASIBLE"
        print(f"Verification: {feasible_str}")
        if result['feasible']:
            print(f"  Verified objective: {result['verified_objective']:.2f}")
            print(f"  Gap: {result['objective_gap']:.2f}")


if __name__ == "__main__":
    main()
