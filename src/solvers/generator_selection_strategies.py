"""
Generator selection strategies for Relax-and-Fix algorithm
"""
import random


def sort_by_power_descending(data):
    """Сортировка по максимальной мощности (убывание) - большие первыми"""
    thermal_gens = data.get("thermal_generators", {})
    gen_power_pairs = [(g, gen_data.get("power_output_maximum", 0.0))
                       for g, gen_data in thermal_gens.items()]
    gen_power_pairs.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in gen_power_pairs]


def sort_by_power_ascending(data):
    """Сортировка по максимальной мощности (возрастание) - маленькие первыми"""
    thermal_gens = data.get("thermal_generators", {})
    gen_power_pairs = [(g, gen_data.get("power_output_maximum", 0.0))
                       for g, gen_data in thermal_gens.items()]
    gen_power_pairs.sort(key=lambda x: x[1], reverse=False)
    return [g for g, _ in gen_power_pairs]


def sort_by_min_production_cost(data):
    """Сортировка по минимальной стоимости производства - дешевые первыми"""
    thermal_gens = data.get("thermal_generators", {})
    gen_cost_pairs = []
    for g, gen_data in thermal_gens.items():
        # Берем первую точку piecewise_production как минимальную стоимость
        pwl = gen_data.get("piecewise_production", [])
        min_cost = pwl[0]["cost"] if pwl else 0.0
        gen_cost_pairs.append((g, min_cost))
    gen_cost_pairs.sort(key=lambda x: x[1], reverse=False)
    return [g for g, _ in gen_cost_pairs]


def sort_by_startup_cost_descending(data):
    """Сортировка по стоимости запуска (убывание) - дорогие запуски первыми"""
    thermal_gens = data.get("thermal_generators", {})
    gen_startup_pairs = []
    for g, gen_data in thermal_gens.items():
        # Берем максимальную стоимость запуска
        startup = gen_data.get("startup", [])
        max_startup_cost = max([s["cost"] for s in startup]) if startup else 0.0
        gen_startup_pairs.append((g, max_startup_cost))
    gen_startup_pairs.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in gen_startup_pairs]


def sort_by_min_ramp_limit(data):
    """Сортировка по минимальному рампу - менее гибкие первыми"""
    thermal_gens = data.get("thermal_generators", {})
    gen_ramp_pairs = []
    for g, gen_data in thermal_gens.items():
        # Берем минимальный из ramp_up и ramp_down
        ramp_up = gen_data.get("ramp_up_limit", float('inf'))
        ramp_down = gen_data.get("ramp_down_limit", float('inf'))
        min_ramp = min(ramp_up, ramp_down)
        gen_ramp_pairs.append((g, min_ramp))
    gen_ramp_pairs.sort(key=lambda x: x[1], reverse=False)
    return [g for g, _ in gen_ramp_pairs]


def sort_by_time_constraints_descending(data):
    """Сортировка по max(time_up_minimum, time_down_minimum) - с большими ограничениями первыми"""
    thermal_gens = data.get("thermal_generators", {})
    gen_time_pairs = []
    for g, gen_data in thermal_gens.items():
        time_up_min = gen_data.get("time_up_minimum", 0)
        time_down_min = gen_data.get("time_down_minimum", 0)
        max_time = max(time_up_min, time_down_min)
        gen_time_pairs.append((g, max_time))
    gen_time_pairs.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in gen_time_pairs]


def sort_must_run_first(data):
    """Must-run генераторы сначала, затем остальные по мощности"""
    thermal_gens = data.get("thermal_generators", {})
    must_run = []
    optional = []

    for g, gen_data in thermal_gens.items():
        power = gen_data.get("power_output_maximum", 0.0)
        if gen_data.get("must_run", 0) == 1:
            must_run.append((g, power))
        else:
            optional.append((g, power))

    # Сортируем обе группы по мощности (убывание)
    must_run.sort(key=lambda x: x[1], reverse=True)
    optional.sort(key=lambda x: x[1], reverse=True)

    return [g for g, _ in must_run] + [g for g, _ in optional]


def sort_by_efficiency(data):
    """Сортировка по эффективности: мощность / минимальная стоимость производства"""
    thermal_gens = data.get("thermal_generators", {})
    gen_efficiency_pairs = []

    for g, gen_data in thermal_gens.items():
        power = gen_data.get("power_output_maximum", 0.0)
        pwl = gen_data.get("piecewise_production", [])
        min_cost = pwl[0]["cost"] if pwl else 1.0  # Избегаем деления на 0
        efficiency = power / min_cost if min_cost > 0 else 0
        gen_efficiency_pairs.append((g, efficiency))

    gen_efficiency_pairs.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in gen_efficiency_pairs]


def sort_random(data):
    """Случайная сортировка (для baseline)"""
    thermal_gens = data.get("thermal_generators", {})
    generators = list(thermal_gens.keys())
    random.shuffle(generators)
    return generators


# Словарь доступных стратегий
STRATEGIES = {
    'power_desc': {
        'name': 'Power (Descending)',
        'description': 'Large generators first',
        'function': sort_by_power_descending
    },
    'power_asc': {
        'name': 'Power (Ascending)',
        'description': 'Small generators first',
        'function': sort_by_power_ascending
    },
    'cheap_production': {
        'name': 'Cheap Production',
        'description': 'Low production cost first',
        'function': sort_by_min_production_cost
    },
    'expensive_startup': {
        'name': 'Expensive Startup',
        'description': 'High startup cost first',
        'function': sort_by_startup_cost_descending
    },
    'inflexible': {
        'name': 'Inflexible First',
        'description': 'Low ramp limits first',
        'function': sort_by_min_ramp_limit
    },
    'time_constrained': {
        'name': 'Time Constrained',
        'description': 'High min up/down time first',
        'function': sort_by_time_constraints_descending
    },
    'must_run_first': {
        'name': 'Must-Run First',
        'description': 'Must-run generators first',
        'function': sort_must_run_first
    },
    'efficiency': {
        'name': 'Efficiency',
        'description': 'Power/cost ratio (high first)',
        'function': sort_by_efficiency
    },
    'random': {
        'name': 'Random',
        'description': 'Random order (baseline)',
        'function': sort_random
    }
}


def get_strategy(strategy_name):
    """Получить стратегию по имени"""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy_name]['function']
