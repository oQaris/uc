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
    # renewable_gens removed

    time_periods = {t + 1: t for t in range(data["time_periods"])}

    gen_startup_categories = {g: list(range(0, len(gen["startup"]))) for (g, gen) in thermal_gens.items()}
    gen_pwl_points = {g: list(range(0, len(gen["piecewise_production"]))) for (g, gen) in thermal_gens.items()}

    # Build model
    m = ConcreteModel()

    # Penalty coefficients for soft constraints (large values to penalize violations)
    DEMAND_PENALTY = 1e6  # $/MW - very high penalty for unmet demand
    RESERVE_PENALTY = 1e5  # $/MW - high penalty for reserve shortfall

    # Keep original instance data for downstream heuristics (sorting generators,
    # sizing lookback for minimum up/down, startup lags, ramps, etc.). This does
    # not change the math model.
    m._uc_data = data

    # Variables
    m.cg = Var(thermal_gens.keys(), time_periods.keys())
    m.pg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.rg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    # m.pw removed (renewable generators)
    m.ug = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)
    m.vg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)
    m.wg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary)

    m.dg = Var(((g, s, t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods),
               within=Binary)
    m.lg = Var(((g, l, t) for g in thermal_gens for l in gen_pwl_points[g] for t in time_periods),
               within=UnitInterval)

    # Slack variables for soft constraints (unmet demand and reserve shortfall)
    m.slack_demand = Var(time_periods.keys(), within=NonNegativeReals)
    m.slack_reserve = Var(time_periods.keys(), within=NonNegativeReals)

    # Objective: minimize generation costs + startup costs + penalties for constraint violations
    m.obj = Objective(expr=
                      # Generation and startup costs
                      sum(
                          sum(
                              m.cg[g, t] + gen["piecewise_production"][0]["cost"] * m.ug[g, t]
                              + sum(
                                  gen_startup["cost"] * m.dg[g, s, t] for (s, gen_startup) in enumerate(gen["startup"]))
                              for t in time_periods)
                          for g, gen in thermal_gens.items())
                      # Penalties for unmet demand and reserve shortfall
                      + sum(DEMAND_PENALTY * m.slack_demand[t] + RESERVE_PENALTY * m.slack_reserve[t]
                            for t in time_periods)
                      )

    # System-wide constraints (soft constraints with slack variables)
    m.demand = Constraint(time_periods.keys())
    m.reserves = Constraint(time_periods.keys())
    for t, t_idx in time_periods.items():
        # Demand constraint: generation + unmet demand >= required demand
        # Unmet demand is penalized heavily in objective
        m.demand[t] = sum(
            m.pg[g, t] + gen["power_output_minimum"] * m.ug[g, t] for (g, gen) in thermal_gens.items()) \
                      + m.slack_demand[t] >= data["demand"][t_idx]
        # Reserve constraint: reserves + shortfall >= required reserves
        # Reserve shortfall is penalized in objective
        m.reserves[t] = sum(m.rg[g, t] for g in thermal_gens) + m.slack_reserve[t] >= data["reserves"][t_idx]

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
    # m.mustrun removed
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
            # m.mustrun removed

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

    # Renewable constraints removed

    return m


if __name__ == "__main__":
    import sys
    import os

    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    print("loading data")
    data = json.load(open(os.path.join(project_root, "examples", "rts_gmlc", "2020-02-09.json")))

    print("building model")
    m = build_uc_model(data)

    print("model setup complete")

    from pyomo.opt import SolverFactory

    solver = SolverFactory("appsi_highs")
    print("solving")
    solver.solve(m, tee=True)

    # print("solving with relax-and-fix")
    # from src.solvers.relax_and_fix import solve_relax_and_fix
    #
    # result = solve_relax_and_fix(
    #     m,
    #     window_size=8,
    #     window_step=8,
    #     gap=0.01,
    #     solver_name="appsi_highs",
    #     verbose=True,
    #     data=data,
    #     model_builder=build_uc_model,
    #     generators_per_iteration=30,
    #     use_limited_horizon=True
    # )
    #
    # print(f"\n=== RESULT ===")
    # print(f"Status: {result['status']}")
    # print(f"Objective: {result['objective']:.2f}")
    # print(f"Solve time: {result['solve_time']:.2f}s")
    # if 'feasible' in result:
    #     print(f"Feasible: {result['feasible']}")
