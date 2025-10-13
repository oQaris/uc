# Test Runner for UC Model

This script runs the Unit Commitment (UC) model on all test instances from the pglib-uc repository and logs the solve times and results.

## Usage

### Basic usage (run on all test instances):
```bash
python run_all_tests.py
```

### Run with custom settings:
```bash
python run_all_tests.py --solver cbc --gap 0.01 --output results.csv
```

### Run on a limited number of instances (for testing):
```bash
python run_all_tests.py --limit 5
```

### Run on specific instances:
```bash
python run_all_tests.py --instances ca/2014-09-01_reserves_0.json ferc/2015-01-01_hw.json
```

### Run with time limit per instance:
```bash
python run_all_tests.py --time-limit 300  # 5 minutes per instance
```

### Run with verbose solver output:
```bash
python run_all_tests.py --verbose
```

## Command-line arguments

- `--solver`: Solver to use (default: cbc)
- `--gap`: MIP gap tolerance (default: 0.01)
- `--time-limit`: Time limit per instance in seconds (default: None)
- `--output`: Output CSV file for results (default: test_results.csv)
- `--verbose`: Show detailed solver output
- `--instances`: List of specific instances to run (default: all)
- `--limit`: Limit the number of instances to run (useful for testing)

## Output

The script generates a CSV file with the following columns:

- **instance**: Name of the test instance file
- **status**: Solver status (optimal, feasible, or error status)
- **solve_time**: Time spent by the solver in seconds
- **build_time**: Time spent building the Pyomo model in seconds
- **load_time**: Time spent loading the JSON data in seconds
- **total_time**: Total time (load + build + solve) in seconds
- **objective_value**: Optimal/best objective value found
- **gap**: MIP gap percentage at termination
- **error**: Error message if the instance failed
- **file_path**: Full path to the instance file

## Test instances

The repository contains test instances in three directories:

1. **ca/** - California ISO test instances (20 files)
   - Various dates with different reserve levels (0%, 1%, 3%, 5%)

2. **ferc/** - FERC test instances (24 files)
   - Monthly data for 2015 with high wind (hw) and low wind (lw) scenarios

3. **rts_gmlc/** - RTS-GMLC test instances (12 files)
   - Monthly data for 2020

Total: **52 test instances**

## Example output

```
Found 52 test instances
Solver: cbc
MIP gap: 0.01

================================================================================
Instance 1/52: ./ca/2014-09-01_reserves_0.json
================================================================================

Loading data from ./ca/2014-09-01_reserves_0.json
  Data loaded in 0.00s
  Building model...
  Model built in 8.83s
  Solving...
  Status: optimal
  Solve time: 106.41s
  Objective: 48256.06

================================================================================
SUMMARY
================================================================================

Results saved to test_results.csv

Total instances: 52
Total time: 5842.15s (97.37 minutes)

Successful solves: 50
Failed solves: 2

Solve time statistics (successful instances):
  Min: 45.23s
  Max: 234.56s
  Avg: 116.84s
  Total: 5842.15s
```

## Requirements

- Python 3.7+
- Pyomo
- CBC solver (or any other MIP solver supported by Pyomo)

Install requirements:
```bash
conda install -c conda-forge pyomo coincbc
```

or

```bash
pip install pyomo
# Install CBC separately from https://github.com/coin-or/Cbc
```

## Notes

- Each instance may take from 1 to 10 minutes to solve depending on size and complexity
- Running all 52 instances may take 1-3 hours total
- Use `--limit` to test on a smaller subset first
- Use `--time-limit` to prevent any single instance from taking too long
- The script is robust to errors and will continue even if some instances fail