"""
Анализ LP файла для выявления причин недостижимости
"""
import re
import json
from collections import defaultdict

def analyze_lp_file(lp_file_path):
    """Анализировать LP файл и найти потенциальные проблемы"""

    print(f"Analyzing LP file: {lp_file_path}")
    print("="*70)

    # Счетчики
    stats = {
        'total_variables': 0,
        'binary_variables': 0,
        'continuous_variables': 0,
        'constraints': 0,
        'demand_constraints': 0,
        'uptime_constraints': 0,
        'downtime_constraints': 0,
    }

    # Информация о demand ограничениях
    demand_info = {}

    # Читаем файл построчно (слишком большой для полной загрузки)
    current_constraint = None
    constraint_rhs = {}

    print("\nPhase 1: Scanning file structure...")

    with open(lp_file_path, 'r') as f:
        line_num = 0
        for line in f:
            line_num += 1
            line = line.strip()

            # Счет ограничений
            if line.startswith('c_'):
                stats['constraints'] += 1
                current_constraint = line.rstrip(':')

                if 'demand' in line:
                    stats['demand_constraints'] += 1
                elif 'uptime' in line:
                    stats['uptime_constraints'] += 1
                elif 'downtime' in line:
                    stats['downtime_constraints'] += 1

            # Найти правую часть ограничения (RHS)
            if line.startswith('=') or line.startswith('>=') or line.startswith('<='):
                if current_constraint:
                    try:
                        value = float(line.split()[1])
                        constraint_rhs[current_constraint] = {
                            'type': line.split()[0],
                            'value': value
                        }
                    except:
                        pass

            # Переменные в секции Binaries
            if line.startswith('ug(') or line.startswith('vg(') or line.startswith('wg(') or line.startswith('dg('):
                stats['binary_variables'] += 1

            if line_num % 100000 == 0:
                print(f"  Processed {line_num} lines...")

    print(f"\nTotal lines processed: {line_num}")

    # Вывод статистики
    print("\n" + "="*70)
    print("LP FILE STATISTICS")
    print("="*70)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Анализ demand ограничений
    print("\n" + "="*70)
    print("DEMAND CONSTRAINTS")
    print("="*70)

    demand_constraints = {k: v for k, v in constraint_rhs.items() if 'demand' in k}
    for const_name, info in sorted(demand_constraints.items()):
        # Извлечь номер периода
        match = re.search(r'demand\((\d+)\)', const_name)
        if match:
            period = int(match.group(1))
            print(f"  Period {period}: demand {info['type']} {info['value']:.2f} MW")

    return stats, constraint_rhs


def check_specific_period_feasibility(lp_file, period, data_file):
    """
    Проверить допустимость для конкретного периода
    """
    import json

    # Загрузить исходные данные
    with open(data_file, 'r') as f:
        data = json.load(f)

    thermal_gens = data.get('thermal_generators', {})
    demand = data.get('demand', [])

    if period >= len(demand):
        print(f"Period {period} is out of range (max {len(demand)-1})")
        return

    period_demand = demand[period]

    print(f"\n{'='*70}")
    print(f"FEASIBILITY CHECK FOR PERIOD {period}")
    print(f"{'='*70}")
    print(f"Demand: {period_demand:.2f} MW")

    # Посчитать максимальную доступную мощность
    total_max_power = sum(gen_data.get('power_output_maximum', 0)
                          for gen_data in thermal_gens.values())

    print(f"Total available capacity (all generators ON): {total_max_power:.2f} MW")

    if total_max_power < period_demand:
        print(f"ERROR: Even with ALL generators ON, cannot meet demand!")
        print(f"Deficit: {period_demand - total_max_power:.2f} MW")
    else:
        print(f"Theoretical surplus: {total_max_power - period_demand:.2f} MW")

    # Теперь найдем какие генераторы зафиксированы на OFF для этого периода
    # Для этого нужно разобрать LP файл более детально
    print(f"\nSearching for fixed variables in period {period}...")

    # Извлечь информацию о ug переменных для этого периода
    pattern = rf'ug\(([^)]+)_{period}\)'

    ug_in_constraints = set()

    with open(lp_file, 'r') as f:
        for line in f:
            matches = re.findall(pattern, line)
            for gen_name in matches:
                ug_in_constraints.add(gen_name)

    print(f"Found {len(ug_in_constraints)} generators mentioned in constraints for period {period}")

    # Сравнить с общим количеством
    total_gens = len(thermal_gens)
    print(f"Total thermal generators in data: {total_gens}")

    if len(ug_in_constraints) < total_gens:
        print(f"WARNING: {total_gens - len(ug_in_constraints)} generators NOT mentioned in constraints!")
        print(f"These generators might be fixed to OFF")


if __name__ == "__main__":
    import sys

    lp_file = r"C:\Users\oQaris\Desktop\Git\uc\debug_models\model_t32-40_g0-61.lp"

    print("LP FILE ANALYZER")
    print("="*70)

    # Базовый анализ
    stats, constraints = analyze_lp_file(lp_file)

    # Попробуем найти файл данных
    print("\n" + "="*70)
    print("Для более детального анализа нужен файл с исходными данными")
    print("Укажите путь к JSON файлу с данными задачи")
    print("Пример: examples/ferc/2020-01-01.json")
    print("="*70)

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        check_specific_period_feasibility(lp_file, 32, data_file)
