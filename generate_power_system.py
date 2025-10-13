import json
import random
from typing import Dict, List, Any
import argparse

def generate_thermal_generator(name: str, generator_type: str, must_run: int = 0) -> Dict[str, Any]:
    """Generate a thermal generator with realistic parameters based on type"""
    
    # Define base parameters for different generator types
    type_params = {
        "STEAM": {
            "power_range": (5, 155),
            "ramp_limits": (20, 60),
            "time_limits": (4, 8),
            "startup_lags": [2, 4, 12],
            "cost_multiplier": 1.0
        },
        "CT": {
            "power_range": (8, 55),
            "ramp_limits": (60, 74),
            "time_limits": (1, 3),
            "startup_lags": [1, 3],
            "cost_multiplier": 0.8
        },
        "CC": {
            "power_range": (170, 355),
            "ramp_limits": (82, 83),
            "time_limits": (5, 8),
            "startup_lags": [5],
            "cost_multiplier": 1.2
        },
        "NUCLEAR": {
            "power_range": (396, 400),
            "ramp_limits": (400, 400),
            "time_limits": (24, 48),
            "startup_lags": [48],
            "cost_multiplier": 0.5
        }
    }
    
    params = type_params[generator_type]
    power_min, power_max = params["power_range"]
    ramp_min, ramp_max = params["ramp_limits"]
    time_min, time_max = params["time_limits"]
    
    # Generate power output (with some randomness within type range)
    actual_power_min = random.uniform(power_min * 0.9, power_min * 1.1)
    actual_power_max = random.uniform(power_max * 0.9, power_max * 1.1)
    
    # Create piecewise production points
    piecewise_production = []
    segments = 4
    for i in range(segments):
        mw = actual_power_min + (actual_power_max - actual_power_min) * (i / (segments - 1))
        base_cost = mw * random.uniform(8, 12) * params["cost_multiplier"]
        cost = base_cost * random.uniform(0.9, 1.1)
        piecewise_production.append({"mw": round(mw, 2), "cost": round(cost, 2)})
    
    # Create startup costs
    startup = []
    for lag in params["startup_lags"]:
        base_startup_cost = actual_power_max * random.uniform(5, 15) * params["cost_multiplier"]
        startup_cost = base_startup_cost * random.uniform(0.8, 1.2)
        startup.append({"lag": lag, "cost": round(startup_cost, 2)})
    
    # Determine initial state
    unit_on_t0 = random.choice([0, 1]) if must_run == 0 else 1
    power_output_t0 = actual_power_min if unit_on_t0 == 1 else 0.0
    time_up_t0 = random.randint(1, 168) if unit_on_t0 == 1 else 0
    time_down_t0 = random.randint(1, 168) if unit_on_t0 == 0 else 0
    time_down_minimum = time_min if (time_max // 2) <= time_min else random.randint(time_min, time_max // 2)
    
    return {
        "must_run": must_run,
        "power_output_minimum": round(actual_power_min, 2),
        "power_output_maximum": round(actual_power_max, 2),
        "ramp_up_limit": round(random.uniform(ramp_min, ramp_max), 2),
        "ramp_down_limit": round(random.uniform(ramp_min, ramp_max), 2),
        "ramp_startup_limit": round(actual_power_min, 2),
        "ramp_shutdown_limit": round(actual_power_min, 2),
        "time_up_minimum": random.randint(time_min, time_max),
        "time_down_minimum": time_down_minimum,
        "power_output_t0": round(power_output_t0, 2),
        "unit_on_t0": unit_on_t0,
        "time_down_t0": time_down_t0,
        "time_up_t0": time_up_t0,
        "startup": startup,
        "piecewise_production": piecewise_production,
        "name": name
    }

def generate_renewable_generator(name: str, resource_type: str, time_periods: int) -> Dict[str, Any]:
    """Generate a renewable generator with time-varying output"""
    
    power_output_minimum = []
    power_output_maximum = []
    
    if resource_type == "PV":
        # Solar PV pattern - zero at night, peaks during day
        for t in range(time_periods):
            hour = t % 24
            if 6 <= hour <= 18:
                min_power = random.uniform(0, 5)
                max_power = random.uniform(20, 100) * random.uniform(0.8, 1.2)
            else:
                min_power = 0.0
                max_power = 0.0
            power_output_minimum.append(round(min_power, 2))
            power_output_maximum.append(round(max_power, 2))
    
    elif resource_type == "WIND":
        # Wind pattern - more variable
        for t in range(time_periods):
            min_power = random.uniform(0, 10)
            max_power = random.uniform(50, 200) * random.uniform(0.7, 1.3)
            power_output_minimum.append(round(min_power, 2))
            power_output_maximum.append(round(max_power, 2))
    
    elif resource_type == "HYDRO":
        # Hydro - more stable but with some variation
        base_power = random.uniform(10, 30)
        for t in range(time_periods):
            variation = random.uniform(0.8, 1.2)
            power = base_power * variation
            power_output_minimum.append(round(power * 0.9, 2))
            power_output_maximum.append(round(power * 1.1, 2))
    
    elif resource_type == "RTPV":  # Rooftop PV
        # Similar to PV but smaller scale
        for t in range(time_periods):
            hour = t % 24
            if 7 <= hour <= 17:
                min_power = random.uniform(0, 2)
                max_power = random.uniform(5, 20) * random.uniform(0.8, 1.2)
            else:
                min_power = 0.0
                max_power = 0.0
            power_output_minimum.append(round(min_power, 2))
            power_output_maximum.append(round(max_power, 2))
    
    else:  # CSP or other
        for t in range(time_periods):
            power_output_minimum.append(0.0)
            power_output_maximum.append(0.0)
    
    return {
        "power_output_minimum": power_output_minimum,
        "power_output_maximum": power_output_maximum,
        "name": name
    }

def generate_demand_and_reserves(time_periods: int) -> tuple:
    """Generate realistic demand and reserve profiles"""
    demand = []
    reserves = []
    
    # Create daily pattern (48 periods = 24 hours * 2)
    for t in range(time_periods):
        hour = t % 24
        
        # Base demand pattern (typical daily curve)
        if 0 <= hour < 6:  # Night
            base_demand = random.uniform(3000, 3300)
        elif 6 <= hour < 9:  # Morning ramp
            base_demand = random.uniform(3300, 3800)
        elif 9 <= hour < 17:  # Day
            base_demand = random.uniform(3800, 4200)
        elif 17 <= hour < 22:  # Evening peak
            base_demand = random.uniform(4200, 4500)
        else:  # Late evening
            base_demand = random.uniform(3500, 3800)
        
        # Add some randomness and trend
        demand_value = base_demand * random.uniform(0.98, 1.02)
        demand.append(round(demand_value, 2))
        
        # Reserves are typically 3-5% of demand
        reserve_value = demand_value * random.uniform(0.03, 0.05)
        reserves.append(round(reserve_value, 2))
    
    return demand, reserves

def generate_power_system_data(
    time_periods: int = 48,
    n_thermal_generators: int = 100,
    n_renewable_generators: int = 50,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate power system data in the specified JSON format
    
    Args:
        time_periods: Number of time periods (default: 48 for half-hourly day)
        n_thermal_generators: Number of thermal generators
        n_renewable_generators: Number of renewable generators
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate demand and reserves
    demand, reserves = generate_demand_and_reserves(time_periods)
    
    # Generate thermal generators
    thermal_generators = {}
    thermal_types = ["STEAM", "CT", "CC", "NUCLEAR"]
    type_weights = [0.3, 0.4, 0.2, 0.1]  # Probability weights
    
    for i in range(n_thermal_generators):
        # Create plant ID (like "101_STEAM_1")
        plant_id = f"{100 + (i % 20):03d}"
        gen_type = random.choices(thermal_types, weights=type_weights)[0]
        unit_num = (i // 20) + 1
        
        name = f"{plant_id}_{gen_type}_{unit_num}"
        
        # Only nuclear plants are must-run
        must_run = 1 if gen_type == "NUCLEAR" else 0
        
        thermal_generators[name] = generate_thermal_generator(name, gen_type, must_run)
    
    # Generate renewable generators
    renewable_generators = {}
    renewable_types = ["PV", "WIND", "HYDRO", "RTPV", "CSP"]
    renewable_weights = [0.3, 0.25, 0.2, 0.2, 0.05]
    
    for i in range(n_renewable_generators):
        plant_id = f"{100 + (i % 20):03d}"
        gen_type = random.choices(renewable_types, weights=renewable_weights)[0]
        unit_num = (i // 20) + 1
        
        name = f"{plant_id}_{gen_type}_{unit_num}"
        renewable_generators[name] = generate_renewable_generator(name, gen_type, time_periods)
    
    return {
        "time_periods": time_periods,
        "demand": demand,
        "reserves": reserves,
        "thermal_generators": thermal_generators,
        "renewable_generators": renewable_generators
    }

def main():
    # parser = argparse.ArgumentParser(description='Generate power system data in JSON format')
    # parser.add_argument('--time_periods', type=int, default=48, help='Number of time periods')
    # parser.add_argument('--thermal_generators', type=int, default=100, help='Number of thermal generators')
    # parser.add_argument('--renewable_generators', type=int, default=50, help='Number of renewable generators')
    # parser.add_argument('--seed', type=int, default=None, help='Random seed')
    # parser.add_argument('--output', type=str, default='generated_power_system.json', help='Output filename')
    
    # args = parser.parse_args()
    
    # Generate data
    data = generate_power_system_data(
        time_periods=48,
        n_thermal_generators=100,
        n_renewable_generators=50,
        seed=42
    )
    
    # Save to file
    with open('generated_power_system.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  - Saved to generated_power_system.json")

if __name__ == "__main__":
    main()