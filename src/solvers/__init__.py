"""
Solvers for Unit Commitment optimization

Available solvers:
- solve_relax_and_fix: Legacy implementation (backward compatible)
- solve_relax_and_fix_v2: Optimized implementation with persistent solver (recommended)
- RelaxAndFixSolver: Class-based API for advanced usage
- solve_ml_assisted: ML-assisted MILP reduction (logistic regression)
"""

# Recommended: Optimized version
from .relax_and_fix_v2 import solve_relax_and_fix_v2, RelaxAndFixSolver

# Legacy: Original implementation
from .relax_and_fix import solve_relax_and_fix

# ML-assisted solver
from .ml_uc_solver import solve_ml_assisted, solve_standard

__all__ = [
    'solve_relax_and_fix_v2',  # Recommended
    'RelaxAndFixSolver',       # Class-based API
    'solve_relax_and_fix',     # Legacy
    'solve_ml_assisted',       # ML-assisted MILP reduction
    'solve_standard',          # Standard MILP baseline
]
