"""
Unit Commitment Model
Reusable function to build UC optimization model from data
"""
import json

from pyomo.environ import ConcreteModel, Var, Objective, Constraint
from pyomo.environ import NonNegativeReals, Binary, UnitInterval


def build_uc_model(data):
    """
    Build Unit Commitment optimization model from data

    Args:
        data: dict containing UC instance data with keys:
            - thermal_generators
            - renewable_generators
            - time_periods
            - demand
            - reserves

    Returns:
        ConcreteModel: Pyomo model ready to be solved
    """
    thermal_gens = data["thermal_generators"]
    renewable_gens = data["renewable_generators"]

    time_periods = {t + 1: t for t in range(data["time_periods"])}

    gen_startup_categories = {g: list(range(0, len(gen["startup"]))) for (g, gen) in thermal_gens.items()}
    gen_pwl_points = {g: list(range(0, len(gen["piecewise_production"]))) for (g, gen) in thermal_gens.items()}

    # Build model
    m = ConcreteModel()

    # Variables
    m.cg = Var(thermal_gens.keys(), time_periods.keys())
    m.pg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.rg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.pw = Var(renewable_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.ug = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)
    m.vg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)
    m.wg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)

    m.dg = Var(((g, s, t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods),
               within=Binary)
    m.lg = Var(((g, l, t) for g in thermal_gens for l in gen_pwl_points[g] for t in time_periods),
               within=UnitInterval)

    # Objective
    m.obj = Objective(expr=sum(
        sum(
            m.cg[g, t] + gen["piecewise_production"][0]["cost"] * m.ug[g, t]
            + sum(gen_startup["cost"] * m.dg[g, s, t] for (s, gen_startup) in enumerate(gen["startup"]))
            for t in time_periods)
        for g, gen in thermal_gens.items())
    )

    # System-wide constraints
    m.demand = Constraint(time_periods.keys())
    m.reserves = Constraint(time_periods.keys())
    for t, t_idx in time_periods.items():
        m.demand[t] = sum(
            m.pg[g, t] + gen["power_output_minimum"] * m.ug[g, t] for (g, gen) in thermal_gens.items()) + sum(
            m.pw[w, t] for w in renewable_gens) == data["demand"][t_idx]
        m.reserves[t] = sum(m.rg[g, t] for g in thermal_gens) >= data["reserves"][t_idx]

    # Initial time constraints
    m.uptimet0 = Constraint(thermal_gens.keys())
    m.downtimet0 = Constraint(thermal_gens.keys())
    m.logicalt0 = Constraint(thermal_gens.keys())
    m.startupt0 = Constraint(thermal_gens.keys())
    m.rampupt0 = Constraint(thermal_gens.keys())
    m.rampdownt0 = Constraint(thermal_gens.keys())
    m.shutdownt0 = Constraint(thermal_gens.keys())

    for g, gen in thermal_gens.items():
        if gen["unit_on_t0"] == 1:
            if gen["time_up_minimum"] - gen["time_up_t0"] >= 1:
                m.uptimet0[g] = sum((m.ug[g, t] - 1) for t in range(1,
                                                                    min(gen["time_up_minimum"] - gen["time_up_t0"],
                                                                        data["time_periods"]) + 1)) == 0
        elif gen["unit_on_t0"] == 0:
            if gen["time_down_minimum"] - gen["time_down_t0"] >= 1:
                m.downtimet0[g] = sum(m.ug[g, t] for t in range(1,
                                                                min(gen["time_down_minimum"] - gen["time_down_t0"],
                                                                    data["time_periods"]) + 1)) == 0
        else:
            raise Exception("Invalid unit_on_t0 for generator {}, unit_on_t0={}".format(g, gen["unit_on_t0"]))

        m.logicalt0[g] = m.ug[g, 1] - gen["unit_on_t0"] == m.vg[g, 1] - m.wg[g, 1]

        startup_expr = sum(
            sum(m.dg[g, s, t]
                for t in range(
                max(1, gen["startup"][s + 1]["lag"] - gen["time_down_t0"] + 1),
                min(gen["startup"][s + 1]["lag"] - 1, data["time_periods"]) + 1
            )
                )
            for s, _ in enumerate(gen["startup"][:-1]))
        if isinstance(startup_expr, int):
            pass
        else:
            m.startupt0[g] = startup_expr == 0

        m.rampupt0[g] = m.pg[g, 1] + m.rg[g, 1] - gen["unit_on_t0"] * (
                gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["ramp_up_limit"]
        m.rampdownt0[g] = gen["unit_on_t0"] * (gen["power_output_t0"] - gen["power_output_minimum"]) - m.pg[g, 1] <= \
                          gen["ramp_down_limit"]

        shutdown_constr = gen["unit_on_t0"] * (gen["power_output_t0"] - gen["power_output_minimum"]) <= gen[
            "unit_on_t0"] * (gen["power_output_maximum"] - gen["power_output_minimum"]) - max(
            (gen["power_output_maximum"] - gen["ramp_shutdown_limit"]), 0) * m.wg[g, 1]

        if isinstance(shutdown_constr, bool):
            pass
        else:
            m.shutdownt0[g] = shutdown_constr

    # Generator constraints
    m.mustrun = Constraint(thermal_gens.keys(), time_periods.keys())
    m.logical = Constraint(thermal_gens.keys(), time_periods.keys())
    m.uptime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.downtime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.startup_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.gen_limit1 = Constraint(thermal_gens.keys(), time_periods.keys())
    m.gen_limit2 = Constraint(thermal_gens.keys(), time_periods.keys())
    m.ramp_up = Constraint(thermal_gens.keys(), time_periods.keys())
    m.ramp_down = Constraint(thermal_gens.keys(), time_periods.keys())
    m.power_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.cost_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.on_select = Constraint(thermal_gens.keys(), time_periods.keys())

    for g, gen in thermal_gens.items():
        for t in time_periods:
            m.mustrun[g, t] = m.ug[g, t] >= gen["must_run"]

            if t > 1:
                m.logical[g, t] = m.ug[g, t] - m.ug[g, t - 1] == m.vg[g, t] - m.wg[g, t]

            UT = min(gen["time_up_minimum"], data["time_periods"])
            if t >= UT:
                m.uptime[g, t] = sum(m.vg[g, i] for i in range(t - UT + 1, t + 1)) <= m.ug[g, t]
            DT = min(gen["time_down_minimum"], data["time_periods"])
            if t >= DT:
                m.downtime[g, t] = sum(m.wg[g, i] for i in range(t - DT + 1, t + 1)) <= 1 - m.ug[g, t]

            m.startup_select[g, t] = m.vg[g, t] == sum(m.dg[g, s, t] for s, _ in enumerate(gen["startup"]))

            m.gen_limit1[g, t] = m.pg[g, t] + m.rg[g, t] <= (
                    gen["power_output_maximum"] - gen["power_output_minimum"]) * m.ug[g, t] - max(
                (gen["power_output_maximum"] - gen["ramp_startup_limit"]), 0) * m.vg[g, t]

            if t < len(time_periods):
                m.gen_limit2[g, t] = m.pg[g, t] + m.rg[g, t] <= (
                        gen["power_output_maximum"] - gen["power_output_minimum"]) * m.ug[g, t] - max(
                    (gen["power_output_maximum"] - gen["ramp_shutdown_limit"]), 0) * m.wg[g, t + 1]

            if t > 1:
                m.ramp_up[g, t] = m.pg[g, t] + m.rg[g, t] - m.pg[g, t - 1] <= gen["ramp_up_limit"]
                m.ramp_down[g, t] = m.pg[g, t - 1] - m.pg[g, t] <= gen["ramp_down_limit"]

            piece_mw1 = gen["piecewise_production"][0]["mw"]
            piece_cost1 = gen["piecewise_production"][0]["cost"]
            m.power_select[g, t] = m.pg[g, t] == sum(
                (piece["mw"] - piece_mw1) * m.lg[g, l, t] for l, piece in enumerate(gen["piecewise_production"]))
            m.cost_select[g, t] = m.cg[g, t] == sum((piece["cost"] - piece_cost1) * m.lg[g, l, t] for l, piece in
                                                    enumerate(gen["piecewise_production"]))
            m.on_select[g, t] = m.ug[g, t] == sum(m.lg[g, l, t] for l, _ in enumerate(gen["piecewise_production"]))

    m.startup_allowed = Constraint(
        ((g, s, t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods))
    for g, gen in thermal_gens.items():
        for s, _ in enumerate(gen["startup"][:-1]):
            for t in time_periods:
                if t >= gen["startup"][s + 1]["lag"]:
                    m.startup_allowed[g, s, t] = m.dg[g, s, t] <= sum(
                        m.wg[g, t - i] for i in range(gen["startup"][s]["lag"], gen["startup"][s + 1]["lag"]))

    # Renewable constraints
    for w, gen in renewable_gens.items():
        for t, t_idx in time_periods.items():
            m.pw[w, t].setlb(gen["power_output_minimum"][t_idx])
            m.pw[w, t].setub(gen["power_output_maximum"][t_idx])

    # keep original instance data for heuristics / sliding-window procedures
    m._uc_data = data
    return m


def _default_generator_sort(data):
    """Sort generators by descending max power (largest first)."""
    thermal = data.get("thermal_generators", {})
    # Use power_output_maximum when scalar, otherwise list -> take max over horizon
    def pmax(gen):
        mx = gen.get("power_output_maximum", 0)
        if isinstance(mx, (list, tuple)):
            return max(mx) if mx else 0
        return mx
    return sorted(list(thermal.keys()), key=lambda g: pmax(thermal[g]), reverse=True)


def _get_thermal_pmax(gen, t_idx=None):
    mx = gen.get("power_output_maximum", 0)
    if isinstance(mx, (list, tuple)):
        if t_idx is None:
            return max(mx) if mx else 0
        return mx[t_idx]
    return mx


def _compute_required_capacity(data, t_idx, include_reserves=True):
    demand = data.get("demand", [])
    reserves = data.get("reserves", [])
    req = demand[t_idx] if t_idx < len(demand) else 0.0
    if include_reserves and t_idx < len(reserves):
        req += reserves[t_idx]
    return req


def _ensure_window_capacity(model, data, gens_sorted, window_start, window_end, current_batch):
    """
    Ensure the chosen batch has enough *potential* capacity to satisfy demand+reserve in the window.
    Expands the batch greedily by adding more generators (from gens_sorted) until feasible by capacity.
    This does NOT guarantee feasibility (ramps/minup etc.), but avoids trivial infeasibility from missing capacity.
    """
    thermal = data.get("thermal_generators", {})
    renew = data.get("renewable_generators", {})
    T = data.get("time_periods", 0)

    chosen = set(current_batch)
    # Precompute renewable max (assume pw is fixed by bounds)
    # In your model, pw has lb/ub equal to min/max, so it can help satisfy demand.
    # We'll assume its upper bound is available.
    def renew_max_at(t_idx):
        tot = 0.0
        for _, wgen in renew.items():
            mx = wgen.get("power_output_maximum", 0)
            if isinstance(mx, (list, tuple)):
                if t_idx < len(mx):
                    tot += mx[t_idx]
            else:
                tot += mx
        return tot

    # Helper: capacity from chosen thermal at time
    def thermal_cap_at(t_idx):
        tot = 0.0
        for g in chosen:
            gen = thermal[g]
            tot += _get_thermal_pmax(gen, t_idx)
        return tot

    # Expand until for all t in window, (renew max + thermal cap) >= demand+reserve
    idx_list = list(range(window_start, window_end))
    # Convert model time (1-based) to data index (0-based)
    data_idxs = [t-1 for t in idx_list if 0 <= t-1 < T]

    pos = 0
    # start from current chosen size position in sorted list
    if gens_sorted:
        # position after last chosen element (approx)
        present = {g:i for i,g in enumerate(gens_sorted) if g in chosen}
        pos = (max(present.values())+1) if present else 0

    while True:
        ok = True
        for t_idx in data_idxs:
            req = _compute_required_capacity(data, t_idx, include_reserves=True)
            avail = renew_max_at(t_idx) + thermal_cap_at(t_idx)
            if avail + 1e-6 < req:
                ok = False
                break
        if ok:
            break
        if pos >= len(gens_sorted):
            # can't expand further
            break
        chosen.add(gens_sorted[pos])
        pos += 1

    return list(chosen)


def _fix_vars_for_non_batch(model, gens, window_start, window_end, batch_set, keep_prev_fixed=True):
    """
    In the current window, unfix binary/dispatch vars for generators in batch_set,
    and fix others to their current values (or 0 if value is None).
    """
    # Vars that are per (g,t)
    g_t_vars = []
    for name in ["ug", "vg", "wg", "pg", "rg", "cg", "dg", "lg"]:
        if hasattr(model, name):
            g_t_vars.append(getattr(model, name))

    # Vars that are (g,s,t)
    g_s_t_vars = []
    if hasattr(model, "delta"):
        g_s_t_vars.append(getattr(model, "delta"))

    for g in gens:
        for t in range(window_start, window_end):
            in_batch = g in batch_set
            for v in g_t_vars:
                if (g, t) not in v:
                    continue
                if in_batch:
                    # allow optimization for batch generators
                    if v[g, t].is_fixed():
                        v[g, t].unfix()
                else:
                    # keep fixed (or fix now)
                    if not v[g, t].is_fixed():
                        val = v[g, t].value
                        if val is None:
                            # safe default: keep commitment OFF unless already on from t0 logic
                            # but don't overwrite existing fixed values
                            val = 0
                        v[g, t].fix(_clamp_value(v[g, t], val) if '_clamp_value' in globals() else val)

            for v in g_s_t_vars:
                # delta[g,s,t]
                for key in list(v.keys()):
                    if len(key)==3 and key[0]==g and key[2]==t:
                        if in_batch:
                            if v[key].is_fixed():
                                v[key].unfix()
                        else:
                            if not v[key].is_fixed():
                                val = v[key].value
                                if val is None:
                                    val = 0
                                v[key].fix(_clamp_value(v[key], val) if '_clamp_value' in globals() else val)


def _activate_time_constraints(model, window_end):
    """
    Treat periods beyond window_end as 'non-existent': deactivate constraints that reference t > window_end.
    For simplicity and safety we deactivate whole indexed components whose index includes time,
    for indices with t > window_end.
    """
    from pyomo.environ import Constraint

    for comp in model.component_objects(ctype=Constraint, active=True):
        # Skip scalar constraints
        if not comp.is_indexed():
            continue
        for idx in comp:
            try:
                # Determine time position: for (t), (g,t), (w,t), (g,s,t)
                t = None
                if isinstance(idx, tuple):
                    if len(idx)==1:
                        t = idx[0]
                    elif len(idx)==2:
                        t = idx[1]
                    elif len(idx)==3:
                        t = idx[2]
                else:
                    t = idx
                if t is not None and t > window_end:
                    comp[idx].deactivate()
            except Exception:
                # If we can't parse the index, leave it active
                pass


def _reactivate_time_constraints(model, window_end):
    """Reactivate constraints with t <= window_end."""
    from pyomo.environ import Constraint

    for comp in model.component_objects(ctype=Constraint):
        if not comp.is_indexed():
            continue
        for idx in comp:
            try:
                t = None
                if isinstance(idx, tuple):
                    if len(idx)==1:
                        t = idx[0]
                    elif len(idx)==2:
                        t = idx[1]
                    elif len(idx)==3:
                        t = idx[2]
                else:
                    t = idx
                if t is not None and t <= window_end:
                    comp[idx].activate()
            except Exception:
                pass



def _clamp_value(var, val, tol=1e-8):
    """Clamp numerical noise to variable domain."""
    if val is None:
        return None
    try:
        dom = var.domain
    except Exception:
        dom = None
    # Binary
    try:
        from pyomo.environ import Binary, UnitInterval, NonNegativeReals
        if dom is Binary:
            if abs(val) <= tol:
                return 0
            if abs(val-1) <= tol:
                return 1
            return int(round(val))
        if dom is UnitInterval:
            if val < 0 and val > -tol:
                val = 0
            if val > 1 and val < 1+tol:
                val = 1
            return min(1, max(0, float(val)))
        if dom is NonNegativeReals:
            if val < 0 and val > -tol:
                val = 0
            return max(0.0, float(val))
    except Exception:
        pass
    # Fallback: just return float
    try:
        return float(val)
    except Exception:
        return val


def _fix_future_periods(model, window_end):
    """Fix all time-indexed variables for periods t > window_end to avoid unboundedness."""
    from pyomo.environ import Var
    for comp in model.component_objects(ctype=Var):
        if not comp.is_indexed():
            continue
        for idx in comp:
            # detect time position in index
            t = None
            if isinstance(idx, tuple):
                if len(idx) == 1:
                    t = idx[0]
                elif len(idx) == 2:
                    t = idx[1]
                elif len(idx) == 3:
                    t = idx[2]
                else:
                    continue
            else:
                t = idx
            if t is None or not isinstance(t, int):
                continue
            if t > window_end:
                v = comp[idx]
                if not v.is_fixed():
                    val = v.value
                    if val is None:
                        # safe defaults by name/domain
                        val = 0
                    v.fix(_clamp_value(v, val))

def _lookback_horizon_for_generator(gen):
    """Conservative lookback for re-unfix to repair feasibility (min up/down, startup lags, ramps)."""
    lb = 0
    lb = max(lb, int(gen.get("time_up_minimum", 0)))
    lb = max(lb, int(gen.get("time_down_minimum", 0)))
    # Startup category lags: gen["startup"] list with {"lag": ...}
    st = gen.get("startup", [])
    for s in st:
        lag = s.get("lag", 0)
        try:
            lb = max(lb, int(lag))
        except Exception:
            pass
    # Ramps can require previous period, keep at least 1
    lb = max(lb, 1)
    return lb


def your_name_func(model, window_size, window_step, gap, verbose=False,
                   generators_per_iteration=None, generator_sort_function=None):
    """
    Solve UC model using Sliding Windows + generator batching (fix/unfix only, hot-start HiGHS).
    Returns the same model with variable values set to the found solution.
    """
    from pyomo.opt import SolverFactory

    if not hasattr(model, "_uc_data"):
        raise ValueError("Model must have model._uc_data with original instance data. "
                         "Call build_uc_model(data) from this module (it sets _uc_data).")

    data = model._uc_data
    thermal = data.get("thermal_generators", {})
    gens_all = list(thermal.keys())
    T = int(data.get("time_periods", 0))

    # Sort generators
    if generator_sort_function is None:
        gens_sorted = _default_generator_sort(data)
    else:
        gens_sorted = list(generator_sort_function(data))

    if generators_per_iteration is None or generators_per_iteration <= 0:
        generators_per_iteration = len(gens_sorted)

    # One solver instance (hot start)
    solver = SolverFactory("appsi_highs")
    solver.config.load_solution = False
    # HiGHS options: ratioGap is supported by appsi_highs in many builds
    options = {
        "ratioGap": float(gap),
        # a few stability/performance options (safe defaults)
        "presolve": "on",
        "mip_heuristic_effort": 0.2,
        "mip_detect_symmetry": "on",
    }

    # Initially, activate all constraints
    _reactivate_time_constraints(model, window_end=T)

    # Main sliding over time windows
    w_start = 1
    last_window_end = 0

    while w_start <= T:
        w_end = min(T, w_start + window_size - 1)

        # Make future periods "non-existent"
        _activate_time_constraints(model, window_end=w_end)
        _fix_future_periods(model, window_end=w_end)

        # Fix all variables beyond w_end to their current values (or 0 if None)
        for vname in ["ug", "vg", "wg", "pg", "rg", "cg", "dg", "lg"]:
            if hasattr(model, vname):
                v = getattr(model, vname)
                for key in v:
                    # key is (g,t)
                    if isinstance(key, tuple) and len(key) == 2:
                        _, t = key
                        if t > w_end and not v[key].is_fixed():
                            val = v[key].value
                            if val is None:
                                val = 0
                            v[key].fix(_clamp_value(v[key], val) if '_clamp_value' in globals() else val)

        if hasattr(model, "delta"):
            v = getattr(model, "delta")
            for key in v:
                if isinstance(key, tuple) and len(key) == 3:
                    _, _, t = key
                    if t > w_end and not v[key].is_fixed():
                        val = v[key].value
                        if val is None:
                            val = 0
                        v[key].fix(_clamp_value(v[key], val) if '_clamp_value' in globals() else val)

        # Generator batching inside the window
        gpos = 0
        while gpos < len(gens_sorted):
            batch = gens_sorted[gpos:gpos + generators_per_iteration]
            # IMPORTANT FIX: ensure enough capacity in the very first batches (and in any window)
            batch = _ensure_window_capacity(model, data, gens_sorted, w_start, w_end + 1, batch)
            batch_set = set(batch)

            # Fix / unfix vars in current window by batch membership
            _fix_vars_for_non_batch(model, gens_all, w_start, w_end + 1, batch_set)

            # Try solving; if infeasible, allow limited "repair unfix" in lookback for batch gens
            solved = False
            max_repairs = 2
            for attempt in range(max_repairs + 1):
                try:
                    res = solver.solve(model, tee=bool(verbose), options=options)
                except RuntimeError as e:
                    # HiGHS appsi can raise if no feasible solution to load
                    solver.config.load_solution = False
                    res = solver.solve(model, tee=bool(verbose), options=options)
                    solver.config.load_solution = True
                term = getattr(res, "termination_condition", None) or getattr(res.solver, "termination_condition", None)
                status = getattr(res, "solver", None)
                # Pyomo appsi returns objects; easiest: check res.solver.termination_condition string
                tc = str(res.solver.termination_condition).lower()
                if "optimal" in tc or "feasible" in tc:
                    solved = True
                    break

                if attempt == max_repairs:
                    break

                # Repair step: unfix a small lookback region for commitment vars for these batch gens
                for g in batch_set:
                    lb = _lookback_horizon_for_generator(thermal[g])
                    t0 = max(1, w_start - lb)
                    t1 = w_end  # only up to current end
                    for t in range(t0, t1 + 1):
                        for vname in ["ug", "vg", "wg"]:
                            if hasattr(model, vname) and (g, t) in getattr(model, vname):
                                vv = getattr(model, vname)[g, t]
                                if vv.is_fixed():
                                    vv.unfix()

            if not solved:
                raise RuntimeError(f"Infeasible subproblem in window [{w_start},{w_end}] for batch starting at {gpos}.")

            # After solve: fix batch vars in window so they won't move later
            for g in batch_set:
                for t in range(w_start, w_end + 1):
                    for vname in ["ug", "vg", "wg", "pg", "rg", "cg", "dg", "lg"]:
                        if hasattr(model, vname) and (g, t) in getattr(model, vname):
                            vv = getattr(model, vname)[g, t]
                            vv.fix(vv.value if vv.value is not None else 0)
                    if hasattr(model, "delta"):
                        v = getattr(model, "delta")
                        for key in list(v.keys()):
                            if len(key)==3 and key[0]==g and key[2]==t:
                                vv = v[key]
                                vv.fix(vv.value if vv.value is not None else 0)

            gpos += generators_per_iteration

        # Move window
        last_window_end = w_end
        w_start += window_step

        # Reactivate constraints up to new end as we progress
        _reactivate_time_constraints(model, window_end=last_window_end)

    # At the end, reactivate all constraints
    _reactivate_time_constraints(model, window_end=T)
    return model

if __name__ == "__main__":
    print("loading data")
    data = json.load(open(r"C:\Users\oQaris\Desktop\Git\uc\examples\ca\2015-03-01_reserves_3.json"))

    print("building model")
    m = build_uc_model(data)

    print("model setup complete")

    from pyomo.opt import SolverFactory

    solver = SolverFactory("appsi_highs")
    solver.config.load_solution = False

    # print("solving")
    # solver.solve(m, options={"ratioGap": 0.01}, tee=True)


    print("solving2")
    your_name_func(
        m,
        window_size=24,
        window_step=6,
        gap=0.01,
        verbose=True,
        generators_per_iteration=None,     # или None
        generator_sort_function=None     # или ваша функция data -> list[g]
    )