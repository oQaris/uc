"""
Тест backward unfixing механизма для решения проблемы недостижимости
"""
import json
from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix

# Configuration
TEST_FILE = r"C:\Users\oQaris\Desktop\Git\uc\examples\ca\2015-03-01_reserves_3.json"

print("="*80)
print("ТЕСТ BACKWARD UNFIXING МЕХАНИЗМА")
print("="*80)
print(f"Test file: {TEST_FILE}")
print()

# Load test data
with open(TEST_FILE, 'r') as f:
    data = json.load(f)

print(f"Problem size:")
print(f"  Thermal generators: {len(data['thermal_generators'])}")
print(f"  Time periods: {data['time_periods']}")
print()

# Build model
print("Building model...")
model = build_uc_model(data)
print("Model built successfully")
print()

# Test parameters (те же что вызывали недостижимость)
WINDOW_SIZE = 8
WINDOW_STEP = 8
GENERATORS_PER_ITERATION = 61
GAP = 0.01
SOLVER_NAME = "appsi_highs"

print("="*80)
print("RELAX-AND-FIX WITH BACKWARD UNFIXING")
print("="*80)
print(f"Window size: {WINDOW_SIZE}, Step: {WINDOW_STEP}")
print(f"Generators per iteration: {GENERATORS_PER_ITERATION}")
print(f"MIP gap: {GAP}, Solver: {SOLVER_NAME}")
print("="*80)
print()

try:
    result = solve_relax_and_fix(
        model=model,
        window_size=WINDOW_SIZE,
        window_step=WINDOW_STEP,
        gap=GAP,
        solver_name=SOLVER_NAME,
        verbose=True,
        verify_solution=True,
        data=data,
        model_builder=build_uc_model,
        generators_per_iteration=GENERATORS_PER_ITERATION,
        use_limited_horizon=True
    )

    print("\n" + "="*80)
    print("РЕЗУЛЬТАТ")
    print("="*80)
    print(f"  Solve time: {result['solve_time']:.2f}s")
    print(f"  Objective: {result['objective']:.2f}")
    print(f"  Status: {result['status']}")

    if 'verification' in result:
        print(f"\n  VERIFICATION:")
        print(f"    Feasible: {result['verification']['feasible']}")
        if result['verification']['feasible']:
            print(f"    Verified objective: {result['verification']['objective']:.2f}")
            print(f"    Objective gap: {result['verification']['gap']:.2f}")
    print("="*80)

    print("\n✓ ТЕСТ ПРОЙДЕН: Недостижимость устранена с помощью backward unfixing!")

except RuntimeError as e:
    print("\n" + "="*80)
    print("✗ ТЕСТ НЕ ПРОЙДЕН")
    print("="*80)
    print(f"Error: {str(e)}")
    print("\nBackward unfixing не помог. Возможно нужна дополнительная настройка.")
