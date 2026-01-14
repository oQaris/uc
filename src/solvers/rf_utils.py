"""Utility functions for Relax-and-Fix algorithm"""
from pyomo.environ import Binary, UnitInterval, Var, value


def find_binary_variables(model):
    """Find all binary variables and their time index positions"""
    binary_vars = []
    for component in model.component_objects(ctype=Var):
        try:
            first_var = next(iter(component.values()))
            if first_var.domain == Binary:
                first_idx = next(iter(component))
                time_idx_pos = 1 if len(first_idx) == 2 else 2  # (g,t) or (g,s,t)
                binary_vars.append((component, time_idx_pos))
        except (StopIteration, ValueError):
            pass
    return binary_vars


def get_generators_sorted_by_power(data):
    """Get generators sorted by max power (descending)"""
    thermal_gens = data.get("thermal_generators", {})
    gen_power_pairs = [(g, gen_data.get("power_output_maximum", 0.0))
                       for g, gen_data in thermal_gens.items()]
    gen_power_pairs.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in gen_power_pairs]


def set_variable_domains(binary_vars, binary_generators, binary_periods):
    """Set variable domains for current subproblem"""
    for var, time_idx_pos in binary_vars:
        for idx in var:
            time_period = idx[time_idx_pos]
            generator = idx[0]

            if generator in binary_generators and time_period in binary_periods:
                var[idx].domain = Binary
            elif not var[idx].is_fixed():
                var[idx].domain = UnitInterval


def fix_variables(binary_vars, fix_generators, fix_periods):
    """Fix variables for current subproblem"""
    for var, time_idx_pos in binary_vars:
        for idx in var:
            generator = idx[0]
            time_period = idx[time_idx_pos]
            if (generator in fix_generators and time_period in fix_periods
                and not var[idx].is_fixed()):
                var[idx].fix()


def unfix_variables_in_window(model, window_periods, binary_vars, verbose=False):
    """Unfix variables in current window"""
    unfixed_count = 0

    for var, time_idx_pos in binary_vars:
        for idx in var:
            time_period = idx[time_idx_pos]
            if time_period in window_periods and var[idx].is_fixed():
                var[idx].unfix()
                unfixed_count += 1

    # Unfix continuous variables
    for var, time_idx_pos in [(model.pg, 1), (model.rg, 1), (model.cg, 1), (model.lg, 2)]:
        for idx in var:
            time_period = idx[time_idx_pos]
            if time_period in window_periods and var[idx].is_fixed():
                var[idx].unfix()
                unfixed_count += 1

    if verbose and unfixed_count > 0:
        print(f"    Unfixed {unfixed_count} variables in window")


def fix_future_variables_to_zero(model, future_periods, binary_vars, generator_lookahead, verbose=False):
    """Fix future variables to zero (selective based on generator lookahead)"""
    fixed_count = 0
    skipped_count = 0

    # Fix binary variables
    for var, time_idx_pos in binary_vars:
        for idx in var:
            generator = idx[0]
            time_period = idx[time_idx_pos]

            if time_period not in future_periods or var[idx].is_fixed():
                continue

            if generator in generator_lookahead and time_period <= generator_lookahead[generator]:
                skipped_count += 1
                continue

            var[idx].fix(0)
            fixed_count += 1

    # Fix continuous variables
    for var, time_idx_pos in [(model.pg, 1), (model.rg, 1), (model.cg, 1), (model.lg, 2)]:
        for idx in var:
            generator = idx[0]
            time_period = idx[time_idx_pos]

            if time_period not in future_periods or var[idx].is_fixed():
                continue

            if generator in generator_lookahead and time_period <= generator_lookahead[generator]:
                skipped_count += 1
                continue

            var[idx].fix(0)
            fixed_count += 1

    if verbose and (fixed_count > 0 or skipped_count > 0):
        print(f"    Fixed {fixed_count} future variables (skipped {skipped_count} critical)")


def refix_periods(model, periods_to_fix, verbose=False):
    """Re-fix all variables in specified periods"""
    if not periods_to_fix:
        return

    fixed_count = 0
    for var_name in ['ug', 'vg', 'wg', 'dg']:
        if not hasattr(model, var_name):
            continue

        var_obj = getattr(model, var_name)
        for idx in var_obj:
            time_period = idx[1] if len(idx) == 2 else idx[2]
            if time_period in periods_to_fix and not var_obj[idx].is_fixed():
                var_obj[idx].fix()
                fixed_count += 1

    if verbose and fixed_count > 0:
        print(f"    Re-fixed {fixed_count} variables in backward-unfixed periods")
