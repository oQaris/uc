# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Unit Commitment (UC) optimization solver based on the Power Grid Lib benchmark library (pglib-uc). The codebase implements a Mixed-Integer Linear Program (MILP) for thermal and renewable generator scheduling to meet electricity demand while minimizing costs.

**Key Features:**
- Solves UC problems with thermal generators (startup costs, ramping, min up/down times)
- Supports piecewise linear production costs and reserves
- Parallel solving of multiple instances
- Performance analysis tools for different datasets

## Project Structure

```
src/
├── models/          # Core optimization model
│   └── uc_model.py  # build_uc_model() - Pyomo model builder
├── runners/         # Test execution scripts
│   └── run_all_tests_parallel.py  # Parallel solver with multiprocessing
└── analisers/       # Analysis and benchmarking tools
    ├── analyze_solver_performance.py  # Statistical analysis
    ├── analyze_instances.py
    └── generate_power_system.py

examples/            # Test instances (JSON format)
├── ca/              # California ISO data (20 instances, ~210s avg solve time)
├── ferc/            # FERC data (24 instances, ~7100s avg solve time)
└── rts_gmlc/        # RTS-GMLC data (12 instances, ~87s avg solve time)
```

## Dependencies

Managed via `pyproject.toml` with Python 3.13+:
- **pyomo** (>=6.9.5) - Optimization modeling framework
- **matplotlib, seaborn** - Visualization
- **pandas, numpy** - Data analysis
- **scikit-learn** - Regression analysis

**Solvers:**
- **HiGHS** (recommended) - Modern, fast MIP solver with excellent performance
- CBC - Alternative open-source solver

**IMPORTANT:** To use HiGHS with Pyomo, you must specify `appsi_highs` in SolverFactory, NOT `highs`:
```python
from pyomo.opt import SolverFactory
solver = SolverFactory("appsi_highs")  # Correct
# solver = SolverFactory("highs")      # Wrong - will not work
```

Install with `uv` (recommended) or pip:
```bash
uv sync
# or
pip install -e .
```

## Common Commands

### Running Tests

**Single instance (for debugging):**
```bash
python src/models/uc_model.py
```

**Parallel execution (recommended for full dataset):**
```bash
# Maximum parallelism on 16-core system (HiGHS recommended)
python src/runners/run_all_tests_parallel.py --parallel 16 --threads-per-instance 1 --solver appsi_highs

# Balanced approach
python src/runners/run_all_tests_parallel.py --parallel 8 --threads-per-instance 2 --solver appsi_highs

# Quick test on subset
python src/runners/run_all_tests_parallel.py --limit 5 --parallel 2 --output quick_test.csv

# With time limit per instance
python src/runners/run_all_tests_parallel.py --parallel 10 --time-limit 300 --solver appsi_highs

# Alternative: using CBC solver
python src/runners/run_all_tests_parallel.py --parallel 16 --solver cbc
```

**Resume functionality:** The parallel runner automatically skips instances already in the output CSV file. Use `--output` with a different filename to re-solve.

### Performance Analysis

After running tests with `--output results.csv`:
```bash
python src/analisers/analyze_solver_performance.py --input results.csv

# Skip plots
python src/analisers/analyze_solver_performance.py --no-plots

# Skip regression
python src/analisers/analyze_solver_performance.py --no-regression
```

## Architecture Details

### UC Model (`src/models/uc_model.py`)

**Function:** `build_uc_model(data) -> ConcreteModel`

**Key Variables:**
- `ug[g,t]` - Binary: unit on/off status
- `vg[g,t]`, `wg[g,t]` - Binary: startup/shutdown indicators
- `pg[g,t]` - Continuous: power output above minimum
- `rg[g,t]` - Continuous: reserves
- `dg[g,s,t]` - Binary: startup category selection
- `lg[g,l,t]` - Continuous: piecewise linear (PWL) segment weights

**Constraint Sets:**
1. System-wide: demand balance, reserve requirements
2. Initial conditions: enforce t0 state (uptime, downtime, ramp from t0)
3. Per-generator per-period: min up/down times, ramping, must-run, power limits
4. Startup logic: startup category selection based on downtime
5. PWL production costs: convex combination constraints

**Input Data Format (JSON):**
```json
{
  "thermal_generators": {"gen_name": {...}},
  "renewable_generators": {"wind_name": {...}},
  "time_periods": 24,
  "demand": [array of length time_periods],
  "reserves": [array of length time_periods]
}
```

### Parallel Runner (`src/runners/run_all_tests_parallel.py`)

**Key Features:**
- Python multiprocessing.Pool for instance-level parallelism
- Thread-safe CSV writing with locks for incremental results
- Metadata extraction (problem size, peak demand, PWL points, etc.)
- Resume functionality via `get_solved_instances()`

**Solver Integration:**
- **HiGHS (appsi_highs):** Uses Pyomo's APPSI (Auto-Persistent Pyomo Solver Interface) for optimal performance
  - Must use `appsi_highs` identifier in SolverFactory, not `highs`
  - Supports native Pyomo integration with options for threads, gap tolerance, time limits
  - Recommended for best performance
- **CBC:** Uses standard Pyomo SolverFactory with options `ratioGap`, `threads`, `seconds`
  - Alternative solver with good stability

**Performance Notes:**
- FERC instances can take 1-5 hours each with 900+ generators
- RTS-GMLC instances solve in 15-200 seconds with ~73 generators
- Optimal parallelism: `--parallel` = number of CPU cores for best throughput

### Analysis Tools

**`analyze_solver_performance.py`:**
- Correlation analysis: identifies `avg_demand` (0.797), `peak_demand` (0.791), `total_pwl_points` (0.747) as strongest predictors
- Generates plots: correlation heatmap, dataset comparison, reserve impact
- Regression models (Linear, Random Forest) for solve time prediction

## Dataset Characteristics

| Dataset  | Instances | Avg Generators | Avg Variables | Avg Solve Time |
|----------|-----------|----------------|---------------|----------------|
| RTS-GMLC | 12        | 73             | 44,544        | ~87s           |
| CA       | 20        | 610            | 305,664       | ~210s          |
| FERC     | 24        | 956            | 477,696       | ~7,100s        |

**Solve Time Drivers:**
1. Number of thermal generators (primary)
2. Average/peak demand
3. PWL complexity (number of points)
4. Reserve requirements

## Development Workflow

When adding new features or modifying the model:

1. **Model Changes:** Edit `src/models/uc_model.py`
   - The model uses Pyomo's concrete model paradigm
   - Constraints are indexed by generators and time periods
   - Test on a small instance (e.g., `examples/rts_gmlc/2020-07-06.json`)

2. **Testing:** Use parallel runner for comprehensive validation
   - Start with `--limit 5` for quick feedback
   - Use `--verbose` for solver output during debugging

3. **Performance Analysis:** Run analyzer after changes
   - Compare solve times before/after modifications
   - Check correlation changes if adding new parameters

4. **Git Workflow:**
   - Repository uses `master` as main branch
   - Commit messages follow project style (see recent commits)
   - Use `uv` for dependency management

## Known Issues and Notes

- **HiGHS Solver:** Must use `appsi_highs` identifier in Pyomo, not `highs` - this is a common mistake
- CBC has limited internal multithreading; instance-level parallelism is more effective
- Windows multiprocessing creates new Python interpreters (overhead); consider Linux for production
- FERC dataset instances may require 5+ hours each; use `--time-limit` to prevent hangs
- The model at line 191 of `uc_model.py` has a hardcoded path in `__main__` for testing
- Default solver in `run_all_tests_parallel.py` is `appsi_highs` (see line 336)

## Performance Optimization Strategies

From analysis reports:

1. **Reduce problem size:** Fewer generators = exponential speedup
2. **Simplify PWL:** Use fewer piecewise linear points
3. **Decrease horizon:** Shorter `time_periods` if feasible
4. **Adjust MIP gap:** Trade solution quality for speed
5. **Use warm starts:** Reuse solutions from similar instances (not yet implemented)

## References

- [UC Model Specification (PDF)](src/models/MODEL.pdf)
- [Detailed Mathematical Model](http://www.optimization-online.org/DB_FILE/2018/11/6930.pdf)
- Open-source implementations: [EGRET](https://github.com/grid-parity-exchange/Egret), [psst](https://github.com/kdheepak/psst)