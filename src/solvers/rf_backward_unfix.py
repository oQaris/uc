"""Backward unfixing for ensuring feasibility across windows"""
from pyomo.environ import Binary, value


def backward_unfix_for_startup(model, window_start, data, verbose=False):
    """
    Unfix minimal set of variables from previous windows to ensure feasibility

    Handles min_uptime/min_downtime constraints by unfixing variables
    from previous periods when needed for generator startup/shutdown
    """
    if window_start <= 1:
        return set()

    thermal_gens = data.get("thermal_generators", {})
    unfixed_periods = set()
    unfixed_count = 0

    for gen_name, gen_data in thermal_gens.items():
        min_downtime = gen_data.get("time_down_minimum", 0)
        min_uptime = gen_data.get("time_up_minimum", 0)
        boundary_period = window_start - 1

        if (gen_name, boundary_period) not in model.ug:
            continue

        ug_boundary = model.ug[gen_name, boundary_period]
        if not ug_boundary.is_fixed():
            continue

        ug_boundary_value = round(value(ug_boundary))

        # Case 1: Generator OFF at boundary, may need to start in new window
        if ug_boundary_value == 0 and min_downtime > 0:
            consecutive_off = _count_consecutive_status(model, gen_name, boundary_period, target_status=0)

            if consecutive_off < min_downtime:
                unfix_start = max(1, window_start - min_downtime)
                unfixed_count += _unfix_generator_vars(model, gen_name, unfix_start, boundary_period, unfixed_periods)

        # Case 2: Generator ON at boundary, may need to stop in new window
        elif ug_boundary_value == 1 and min_uptime > 0:
            consecutive_on = _count_consecutive_status(model, gen_name, boundary_period, target_status=1)

            if consecutive_on < min_uptime:
                unfix_start = max(1, window_start - min_uptime)
                unfixed_count += _unfix_generator_vars(model, gen_name, unfix_start, boundary_period, unfixed_periods)

    if verbose and unfixed_count > 0:
        unique_periods = sorted(unfixed_periods)
        print(f"    Backward unfixing: {unfixed_count} vars in {len(unique_periods)} periods "
              f"({min(unique_periods)}-{max(unique_periods)})")

    return unfixed_periods


def _count_consecutive_status(model, gen_name, start_period, target_status):
    """Count how many consecutive periods generator had target_status"""
    consecutive = 1
    for t in range(start_period - 1, 0, -1):
        if (gen_name, t) in model.ug and model.ug[gen_name, t].is_fixed():
            if round(value(model.ug[gen_name, t])) == target_status:
                consecutive += 1
            else:
                break
        else:
            break
    return consecutive


def _unfix_generator_vars(model, gen_name, start_period, end_period, unfixed_periods_set):
    """Unfix ug, vg, wg for generator in period range"""
    count = 0
    for t in range(start_period, end_period + 1):
        for var_name in ['ug', 'vg', 'wg']:
            if hasattr(model, var_name) and (gen_name, t) in getattr(model, var_name):
                var = getattr(model, var_name)[gen_name, t]
                if var.is_fixed():
                    var.unfix()
                    var.domain = Binary
                    count += 1
                    unfixed_periods_set.add(t)
    return count
