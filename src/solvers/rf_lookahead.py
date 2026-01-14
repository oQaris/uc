"""Adaptive lookahead and constraint management for Relax-and-Fix"""
from pyomo.environ import value


def calculate_generator_lookahead(model, boundary_period, fix_end_period, data, num_periods):
    """Calculate individual lookahead for each generator based on state at boundary"""
    thermal_gens = data.get("thermal_generators", {})
    generator_lookahead = {}

    min_lookahead_end = min(fix_end_period + 1, num_periods)
    remaining_horizon = num_periods - fix_end_period

    # Calculate max constraint across all generators
    max_constraint_global = max(
        max(gen_data.get("time_up_minimum", 0), gen_data.get("time_down_minimum", 0))
        for gen_data in thermal_gens.values()
    ) if thermal_gens else 0

    use_full_horizon = remaining_horizon <= max_constraint_global

    for g, gen_data in thermal_gens.items():
        min_uptime = gen_data.get("time_up_minimum", 0)
        min_downtime = gen_data.get("time_down_minimum", 0)

        if use_full_horizon:
            generator_lookahead[g] = num_periods
        else:
            max_constraint = max(min_uptime, min_downtime)
            constraint_based = min(fix_end_period + max_constraint, num_periods) if max_constraint > 0 else min_lookahead_end
            min_reasonable = min(fix_end_period + remaining_horizon // 2, num_periods)
            generator_lookahead[g] = max(constraint_based, min_reasonable)

        # Adjust based on boundary state
        if boundary_period == 0:
            ug_at_boundary = 1 if gen_data.get("initial_status", 0) > 0 else 0
        else:
            try:
                ug_at_boundary = round(value(model.ug[g, boundary_period - 1]))
            except:
                ug_at_boundary = 0

        # Check if generator needs extended lookahead
        if ug_at_boundary == 1 and min_uptime > 0:
            uptime_so_far = _count_uptime(model, g, boundary_period)
            if uptime_so_far < min_uptime:
                required_end = min(fix_end_period + (min_uptime - uptime_so_far), num_periods)
                generator_lookahead[g] = max(generator_lookahead[g], required_end)

        elif ug_at_boundary == 0 and min_downtime > 0:
            downtime_so_far = _count_downtime(model, g, boundary_period)
            if downtime_so_far < min_downtime:
                required_end = min(fix_end_period + (min_downtime - downtime_so_far), num_periods)
                generator_lookahead[g] = max(generator_lookahead[g], required_end)

    return generator_lookahead


def manage_constraints_for_window(model, generator_lookahead, num_periods, verbose=False):
    """Activate/deactivate constraints based on individual generator lookahead"""
    max_lookahead = max(generator_lookahead.values()) if generator_lookahead else num_periods

    system_constraints = [(model.demand, 0), (model.reserves, 0)]
    generator_constraints = [
        (model.mustrun, 1, 0), (model.logical, 1, 0), (model.uptime, 1, 0),
        (model.downtime, 1, 0), (model.startup_select, 1, 0), (model.gen_limit1, 1, 0),
        (model.gen_limit2, 1, 0), (model.ramp_up, 1, 0), (model.ramp_down, 1, 0),
        (model.power_select, 1, 0), (model.cost_select, 1, 0), (model.on_select, 1, 0),
        (model.startup_allowed, 2, 0)
    ]

    deactivated, activated = 0, 0

    # System constraints active up to max_lookahead
    for constraint, time_idx_pos in system_constraints:
        for idx in constraint:
            should_activate = idx <= max_lookahead
            if should_activate and not constraint[idx].active:
                constraint[idx].activate()
                activated += 1
            elif not should_activate and constraint[idx].active:
                constraint[idx].deactivate()
                deactivated += 1

    # Generator constraints active individually
    for constraint, time_idx_pos, gen_idx_pos in generator_constraints:
        for idx in constraint:
            if not isinstance(idx, tuple):
                continue

            generator = idx[gen_idx_pos]
            time_period = idx[time_idx_pos]
            required_end = generator_lookahead.get(generator, max_lookahead)
            should_activate = time_period <= required_end

            if should_activate and not constraint[idx].active:
                constraint[idx].activate()
                activated += 1
            elif not should_activate and constraint[idx].active:
                constraint[idx].deactivate()
                deactivated += 1

    if verbose and (deactivated > 0 or activated > 0):
        print(f"    Constraints: activated {activated}, deactivated {deactivated}")


def _count_uptime(model, g, boundary_period):
    """Count consecutive uptime before boundary"""
    uptime = 1
    for t in range(boundary_period - 2, -1, -1):
        try:
            if round(value(model.ug[g, t])) == 1:
                uptime += 1
            else:
                break
        except:
            break
    return uptime


def _count_downtime(model, g, boundary_period):
    """Count consecutive downtime before boundary"""
    downtime = 1
    for t in range(boundary_period - 2, -1, -1):
        try:
            if round(value(model.ug[g, t])) == 0:
                downtime += 1
            else:
                break
        except:
            break
    return downtime
