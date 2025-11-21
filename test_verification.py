"""
Test clean verification of Relax-and-Fix solution
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix


def main():
    # Test on RTS-GMLC dataset example (smaller, faster)
    DATA_FILE = r"C:\Users\oQaris\Desktop\Git\uc\examples\rts_gmlc\2020-07-06.json"
    SOLVER = "appsi_highs"
    GAP = 0.0
    VERBOSE = True

    print("="*80)
    print("TESTING CLEAN VERIFICATION OF RELAX-AND-FIX SOLUTION")
    print("="*80)
    print(f"\nData file: {DATA_FILE}")
    print(f"Solver: {SOLVER}")
    print(f"Gap: {GAP}")

    # Load data
    print("\nLoading data...")
    with open(DATA_FILE) as f:
        data = json.load(f)

    print(f"Instance: {data['time_periods']} periods, {len(data['thermal_generators'])} thermal gens")

    # Build model
    print("\nBuilding model...")
    model = build_uc_model(data)

    # Solve with Relax-and-Fix
    print("\n" + "="*80)
    print("SOLVING WITH RELAX-AND-FIX")
    print("="*80)

    result = solve_relax_and_fix(
        model=model,
        window_size=8,
        window_step=6,
        solver_name=SOLVER,
        gap=GAP,
        verbose=VERBOSE,
        verify_solution=True,
        data=data,
        model_builder=build_uc_model
    )

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSolve time: {result['solve_time']:.2f}s")
    print(f"Relax-and-fix objective: {result['objective']:,.2f}")
    print(f"Rounded variables: {result['rounded_variables']}")

    if 'feasible' in result:
        print(f"\nVerification in fresh model:")
        if result['feasible']:
            print(f"  Status: FEASIBLE")
            print(f"  Verified objective: {result['verified_objective']:,.2f}")
            print(f"  Absolute gap: {result['objective_gap']:,.2f}")
            print(f"  Relative gap: {(result['objective_gap']/result['objective'])*100:.4f}%")

            if result['objective_gap'] < 1e-3:
                print(f"\n  Result: Solution is VALID and objectives match perfectly!")
            else:
                print(f"\n  Result: Solution is VALID but objectives differ slightly")
                print(f"          (This can happen due to numerical precision or rounding)")
        else:
            print(f"  Status: INFEASIBLE")
            print(f"\n  ERROR: Relax-and-fix found an INVALID solution!")
            print(f"         This should not happen and indicates a bug.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
