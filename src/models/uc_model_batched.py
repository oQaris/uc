"""
Unit Commitment Model
Reusable function to build UC optimization model from data
"""
import json

from pyomo.environ import ConcreteModel, Var, Objective, Constraint
from pyomo.environ import NonNegativeReals, Binary, UnitInterval
from copy import deepcopy
from typing import Dict, Any, Optional
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.environ import value

from src.solvers.relax_and_fix import _verify_solution_feasibility


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

    return m


# =========================
# Rolling-window / generator-batching decomposition
# =========================
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any

def _chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def _slice_list(a: List[float], t_start: int, t_end: int) -> List[float]:
    # t_start/t_end are 1-based inclusive
    return a[t_start-1:t_end]

def build_uc_subproblem(
    data_full: Dict[str, Any],
    thermal_subset: List[str],
    t_start: int,
    t_end: int,
    init_state: Dict[str, Dict[str, float]],
    fixed_generation: List[float],
    fixed_reserves: List[float],
    slack_penalties: Optional[Dict[str, float]] = None,
):
    """Build a *small* UC model for a subset of thermal generators on a time window.

    The system-wide balance/reserve constraints include constant offsets coming from
    generators that are *not* in thermal_subset (they are treated as fixed).

    To keep every subproblem feasible even with imperfect fixing, we add small
    nonnegative slack variables:
      - load_shed[t]   : unmet demand (penalized heavily)
      - spill[t]       : excess generation dumped (penalized)
      - reserve_short[t]: reserve shortage (penalized)

    Args:
      data_full: original instance dict (same schema as build_uc_model)
      thermal_subset: list of thermal generator IDs to keep as decision vars
      t_start, t_end: 1-based inclusive window in the original horizon
      init_state: per-generator initial state to be used as t0 for this window:
          {g: {"unit_on_t0":0/1, "time_up_t0":int, "time_down_t0":int, "power_output_t0":float}}
      fixed_generation: list length H with fixed thermal generation (MW) from excluded units
      fixed_reserves: list length H with fixed reserves (MW) from excluded units
      slack_penalties: {"shed":..., "spill":..., "reserve_short":...}

    Returns:
      (model, window_data) where window_data includes mapping back to original indices.
    """
    if slack_penalties is None:
        slack_penalties = {"shed": 1e8, "spill": 1e4, "reserve_short": 1e7}

    H = t_end - t_start + 1
    assert len(fixed_generation) == H
    assert len(fixed_reserves) == H

    # --- Slice data to the window and subset ---
    data = {
        "time_periods": H,
        "demand": _slice_list(data_full["demand"], t_start, t_end),
        "reserves": _slice_list(data_full["reserves"], t_start, t_end),
        "thermal_generators": {g: deepcopy(data_full["thermal_generators"][g]) for g in thermal_subset},
        "renewable_generators": {},
    }
    for w, gen in data_full["renewable_generators"].items():
        gen2 = deepcopy(gen)
        gen2["power_output_minimum"] = _slice_list(gen["power_output_minimum"], t_start, t_end)
        gen2["power_output_maximum"] = _slice_list(gen["power_output_maximum"], t_start, t_end)
        data["renewable_generators"][w] = gen2

    # Override t0 state for selected thermals (critical for min up/down + ramps)
    for g in thermal_subset:
        st = init_state[g]
        data["thermal_generators"][g]["unit_on_t0"] = int(st["unit_on_t0"])
        data["thermal_generators"][g]["time_up_t0"] = int(st["time_up_t0"])
        data["thermal_generators"][g]["time_down_t0"] = int(st["time_down_t0"])
        data["thermal_generators"][g]["power_output_t0"] = float(st["power_output_t0"])

    # --- Build model (mostly same as build_uc_model, but with offsets + slacks) ---
    thermal_gens = data["thermal_generators"]
    renewable_gens = data["renewable_generators"]
    time_periods = {t + 1: t for t in range(data["time_periods"])}  # 1..H -> 0..H-1

    gen_startup_categories = {g: list(range(0, len(gen["startup"]))) for (g, gen) in thermal_gens.items()}
    gen_pwl_points = {g: list(range(0, len(gen["piecewise_production"]))) for (g, gen) in thermal_gens.items()}

    m = ConcreteModel()

    # Variables (same core vars, plus small slacks)
    m.cg = Var(thermal_gens.keys(), time_periods.keys())
    m.pg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.rg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.pw = Var(renewable_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.ug = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)
    m.vg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)
    m.wg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)
    m.dg = Var(((g, s, t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods), within=Binary)
    m.lg = Var(((g, l, t) for g in thermal_gens for l in gen_pwl_points[g] for t in time_periods), within=UnitInterval)

    m.load_shed = Var(time_periods.keys(), within=NonNegativeReals)
    m.spill = Var(time_periods.keys(), within=NonNegativeReals)
    m.reserve_short = Var(time_periods.keys(), within=NonNegativeReals)

    # Objective
    m.obj = Objective(expr=sum(
        sum(
            m.cg[g, t] + gen["piecewise_production"][0]["cost"] * m.ug[g, t]
            + sum(gen_startup["cost"] * m.dg[g, s, t] for (s, gen_startup) in enumerate(gen["startup"]))
            for t in time_periods
        )
        for g, gen in thermal_gens.items()
    ) + slack_penalties["shed"] * sum(m.load_shed[t] for t in time_periods)
      + slack_penalties["spill"] * sum(m.spill[t] for t in time_periods)
      + slack_penalties["reserve_short"] * sum(m.reserve_short[t] for t in time_periods)
    )

    # System-wide constraints (with fixed offsets + slacks)
    m.demand = Constraint(time_periods.keys())
    m.reserves = Constraint(time_periods.keys())
    for t, t_idx in time_periods.items():
        thermal_expr = sum(
            m.pg[g, t] + gen["power_output_minimum"] * m.ug[g, t] for (g, gen) in thermal_gens.items()
        )
        renew_expr = sum(m.pw[w, t] for w in renewable_gens)
        m.demand[t] = thermal_expr + renew_expr + fixed_generation[t_idx] + m.load_shed[t] - m.spill[t] == data["demand"][t_idx]
        m.reserves[t] = sum(m.rg[g, t] for g in thermal_gens) + fixed_reserves[t_idx] + m.reserve_short[t] >= data["reserves"][t_idx]

    # Initial time constraints (same logic as build_uc_model)
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
                m.uptimet0[g] = sum((m.ug[g, t] - 1) for t in range(
                    1, min(gen["time_up_minimum"] - gen["time_up_t0"], data["time_periods"]) + 1
                )) == 0
        elif gen["unit_on_t0"] == 0:
            if gen["time_down_minimum"] - gen["time_down_t0"] >= 1:
                m.downtimet0[g] = sum(m.ug[g, t] for t in range(
                    1, min(gen["time_down_minimum"] - gen["time_down_t0"], data["time_periods"]) + 1
                )) == 0
        else:
            raise Exception(f"Invalid unit_on_t0 for generator {g}, unit_on_t0={gen['unit_on_t0']}")

        m.logicalt0[g] = m.ug[g, 1] - gen["unit_on_t0"] == m.vg[g, 1] - m.wg[g, 1]

        startup_expr = sum(
            sum(m.dg[g, s, t] for t in range(
                max(1, gen["startup"][s + 1]["lag"] - gen["time_down_t0"] + 1),
                min(gen["startup"][s + 1]["lag"] - 1, data["time_periods"]) + 1
            ))
            for s, _ in enumerate(gen["startup"][:-1])
        )
        if not isinstance(startup_expr, int):
            m.startupt0[g] = startup_expr == 0

        m.rampupt0[g] = m.pg[g, 1] + m.rg[g, 1] - gen["unit_on_t0"] * (
            gen["power_output_t0"] - gen["power_output_minimum"]
        ) <= gen["ramp_up_limit"]

        m.rampdownt0[g] = gen["unit_on_t0"] * (gen["power_output_t0"] - gen["power_output_minimum"]) - m.pg[g, 1] <= gen["ramp_down_limit"]

        shutdown_constr = gen["unit_on_t0"] * (gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["unit_on_t0"] * (
            gen["power_output_maximum"] - gen["power_output_minimum"]
        ) - max((gen["power_output_maximum"] - gen["ramp_shutdown_limit"]), 0) * m.wg[g, 1]
        if not isinstance(shutdown_constr, bool):
            m.shutdownt0[g] = shutdown_constr

    # Generator constraints (same as build_uc_model, incl. fixed uptime/downtime summations)
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
                gen["power_output_maximum"] - gen["power_output_minimum"]
            ) * m.ug[g, t] - max((gen["power_output_maximum"] - gen["ramp_startup_limit"]), 0) * m.vg[g, t]

            if t < len(time_periods):
                m.gen_limit2[g, t] = m.pg[g, t] + m.rg[g, t] <= (
                    gen["power_output_maximum"] - gen["power_output_minimum"]
                ) * m.ug[g, t] - max((gen["power_output_maximum"] - gen["ramp_shutdown_limit"]), 0) * m.wg[g, t + 1]

            if t > 1:
                m.ramp_up[g, t] = m.pg[g, t] + m.rg[g, t] - m.pg[g, t - 1] <= gen["ramp_up_limit"]
                m.ramp_down[g, t] = m.pg[g, t - 1] - m.pg[g, t] <= gen["ramp_down_limit"]

            piece_mw1 = gen["piecewise_production"][0]["mw"]
            piece_cost1 = gen["piecewise_production"][0]["cost"]
            m.power_select[g, t] = m.pg[g, t] == sum((piece["mw"] - piece_mw1) * m.lg[g, l, t] for l, piece in enumerate(gen["piecewise_production"]))
            m.cost_select[g, t] = m.cg[g, t] == sum((piece["cost"] - piece_cost1) * m.lg[g, l, t] for l, piece in enumerate(gen["piecewise_production"]))
            m.on_select[g, t] = m.ug[g, t] == sum(m.lg[g, l, t] for l, _ in enumerate(gen["piecewise_production"]))

    m.startup_allowed = Constraint(((g, s, t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods))
    for g, gen in thermal_gens.items():
        for s, _ in enumerate(gen["startup"][:-1]):
            for t in time_periods:
                if t >= gen["startup"][s + 1]["lag"]:
                    m.startup_allowed[g, s, t] = m.dg[g, s, t] <= sum(
                        m.wg[g, t - i] for i in range(gen["startup"][s]["lag"], gen["startup"][s + 1]["lag"])
                    )

    # Renewable bounds
    for w, gen in renewable_gens.items():
        for t, t_idx in time_periods.items():
            m.pw[w, t].setlb(gen["power_output_minimum"][t_idx])
            m.pw[w, t].setub(gen["power_output_maximum"][t_idx])

    window_meta = {"t_start": t_start, "t_end": t_end, "H": H}
    return m, window_meta


class RollingBatchedUCSolver:
    """Heuristic decomposition:
      - rolling (possibly overlapping) time windows
      - within each window: solve batches of N thermal generators sequentially (Gauss-Seidel)

    This is *not* guaranteed optimal, but it typically reduces wall-clock time a lot,
    and it keeps each MILP small by construction.
    """

    def __init__(self, data_full: Dict[str, Any], solver_name: str = "appsi_highs", solver_options: Optional[Dict[str, Any]] = None):
        self.data = data_full
        self.solver_name = solver_name
        self.solver_options = solver_options or {}

    def solve(
        self,
        batch_size: int,
        window: int,
        step: Optional[int] = None,
        lookahead: Optional[int] = None,
        batch_iters: int = 1,
        slack_penalties: Optional[Dict[str, float]] = None,
        tee: bool = False,
    ) -> Dict[str, Any]:
        """Solve with rolling windows and generator batching.

        Args:
          batch_size: N generators per MILP
          window: base window length (hours)
          step: how many hours to *commit/fix* and advance each iteration (default=window)
          lookahead: extra overlap on the right to protect min up/down (default=max(UT,DT) over all gens)
          batch_iters: number of Gauss-Seidel passes over batches per window (>=1)
        """
        from pyomo.opt import SolverFactory

        thermal_ids = list(self.data["thermal_generators"].keys())
        renewable_ids = list(self.data["renewable_generators"].keys())
        T = int(self.data["time_periods"])

        if step is None:
            step = window

        if lookahead is None:
            lookahead = 0
            for g, gen in self.data["thermal_generators"].items():
                lookahead = max(lookahead, int(gen["time_up_minimum"]), int(gen["time_down_minimum"]))

        # --- Current schedule estimate (will be updated) ---
        # 1-based time indexing for convenience.
        u = {g: [None] + [int(self.data["thermal_generators"][g]["must_run"] or 0)] * T for g in thermal_ids}
        p_total = {g: [None] + [self.data["thermal_generators"][g]["power_output_minimum"] * u[g][t] for t in range(1, T+1)] for g in thermal_ids}
        r = {g: [None] + [0.0] * T for g in thermal_ids}
        v = {g: [None] + [0] * T for g in thermal_ids}
        w = {g: [None] + [0] * T for g in thermal_ids}
        pw = {wid: [None] + [0.0] * T for wid in renewable_ids}

        # --- State carried between windows (used as t0 in subproblems) ---
        state = {}
        for g, gen in self.data["thermal_generators"].items():
            state[g] = {
                "unit_on_t0": int(gen["unit_on_t0"]),
                "time_up_t0": int(gen["time_up_t0"]),
                "time_down_t0": int(gen["time_down_t0"]),
                "power_output_t0": float(gen["power_output_t0"]),
            }

        batches = _chunked(thermal_ids, batch_size)

        # Rolling horizon
        start = 1
        while start <= T:
            commit_end = min(T, start + step - 1)
            solve_end = min(T, start + window + lookahead - 1)

            for _it in range(batch_iters):
                for batch in batches:
                    # fixed contribution of excluded thermal units (based on current estimate)
                    fixed_gen = []
                    fixed_res = []
                    for t_glob in range(start, solve_end + 1):
                        fg = 0.0
                        fr = 0.0
                        for g in thermal_ids:
                            if g in batch:
                                continue
                            fg += float(p_total[g][t_glob])
                            fr += float(r[g][t_glob])
                        fixed_gen.append(fg)
                        fixed_res.append(fr)

                    # init_state for this batch at (start-1)
                    init_state = {g: deepcopy(state[g]) for g in batch}

                    m, _meta = build_uc_subproblem(
                        data_full=self.data,
                        thermal_subset=batch,
                        t_start=start,
                        t_end=solve_end,
                        init_state=init_state,
                        fixed_generation=fixed_gen,
                        fixed_reserves=fixed_res,
                        slack_penalties=slack_penalties,
                    )

                    solver = SolverFactory(self.solver_name)
                    if hasattr(solver, 'config'):
                        solver.config.mip_gap = self.solver_options['ratioGap']
                    res = solver.solve(m, tee=tee, options=self.solver_options)

                    # Update schedule estimate for this batch on the solved horizon
                    for t_loc, t_idx in {t+1: t for t in range(solve_end - start + 1)}.items():
                        t_glob = start + t_idx
                        for g in batch:
                            ug = int(round(m.ug[g, t_loc].value))
                            u[g][t_glob] = ug
                            pg = float(m.pg[g, t_loc].value or 0.0)
                            rg = float(m.rg[g, t_loc].value or 0.0)
                            p_total[g][t_glob] = pg + self.data["thermal_generators"][g]["power_output_minimum"] * ug
                            r[g][t_glob] = rg
                            v[g][t_glob] = int(round(m.vg[g, t_loc].value or 0.0))
                            w[g][t_glob] = int(round(m.wg[g, t_loc].value or 0.0))

                        for wid in renewable_ids:
                            if wid in m.pw:
                                pw[wid][t_glob] = float(m.pw[wid, t_loc].value or 0.0)

            # Freeze [start..commit_end] and advance state
            for t_glob in range(start, commit_end + 1):
                for g in thermal_ids:
                    ug = int(u[g][t_glob])
                    state_g = state[g]
                    prev_u = int(state_g["unit_on_t0"])
                    if ug == 1:
                        state_g["time_up_t0"] = (state_g["time_up_t0"] + 1) if prev_u == 1 else 1
                        state_g["time_down_t0"] = 0
                    else:
                        state_g["time_down_t0"] = (state_g["time_down_t0"] + 1) if prev_u == 0 else 1
                        state_g["time_up_t0"] = 0
                    state_g["unit_on_t0"] = ug
                    state_g["power_output_t0"] = float(p_total[g][t_glob])

            start = commit_end + 1


        # ---------------------------
        # POST: Build full original model and impose the schedule
        # ---------------------------
        full = build_uc_model(self.data)

        thermal_ids = list(self.data["thermal_generators"].keys())
        T = int(self.data["time_periods"])

        # 1) поставить ug и зафиксировать (это главное решение эвристики)
        for g in thermal_ids:
            for t in range(1, T + 1):
                val_u = int(u[g][t])
                full.ug[g, t].value = val_u
                full.ug[g, t].fix(val_u)

        # (опционально) можно подсунуть initial guess для pw/pg/rg если хочешь
        # но фиксировать их обычно НЕ нужно:
        # for wid in renewable_ids:
        #     for t in range(1, T+1):
        #         full.pw[wid, t].value = float(pw[wid][t])

        # 2) решить модель при фиксированных ug (vg/wg/dg станут согласованными)
        solver = SolverFactory(self.solver_name)

        # аккуратно с mip_gap: у тебя раньше было self.solver_options['ratioGap'] (может KeyError)
        mip_gap = self.solver_options.get("ratioGap", None)

        solve_kwargs = dict(tee=tee)
        solve_kwargs["symbolic_solver_labels"] = True

        if hasattr(solver, "config"):
            # APPSI
            if mip_gap is not None:
                solver.config.mip_gap = mip_gap
            res = solver.solve(full, **solve_kwargs)
        else:
            # legacy
            opts = dict(self.solver_options)
            if mip_gap is not None:
                opts["ratioGap"] = mip_gap
            res = solver.solve(full, options=opts, **solve_kwargs)

        if res.solver.termination_condition != TerminationCondition.optimal:
            # можно либо падать, либо вернуть модель как есть
            raise RuntimeError(f"Full-model completion failed: {res.solver.termination_condition}")

        # Теперь full содержит ВСЕ переменные (включая dg/vg/wg) с валидными значениями
        return full


def main():
    data = json.load(open(r"C:\Users\oQaris\Desktop\Git\uc\examples\ca\2014-09-01_reserves_1.json"))

    solver = RollingBatchedUCSolver(
        data,
        solver_name="appsi_highs",          # или другой MILP солвер
        solver_options={"ratioGap": 0.01},  # зависит от солвера
    )

    sol = solver.solve(
        batch_size=10,   # N генераторов в пачке
        window=24,       # основное окно (часы)
        step=6,          # сколько часов фиксируем и сдвигаемся
        lookahead=None,  # None => max(UT,DT) автоматически
        batch_iters=2,   # можно 2-3 для лучшей согласованности пачек
        tee=True,
    )

    check = _verify_solution_feasibility(
        old_model=sol,
        data=data,
        model_builder=build_uc_model,
        solver_name="appsi_highs",
        gap=0.0,
        verbose=True,
    )
    print(check)


if __name__ == "__main__":
    main()