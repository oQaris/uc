"""
Детальный анализ проблемы недостижимости на периоде 32
"""
import json
from pyomo.environ import value

# Загрузка данных
data_file = r"C:\Users\oQaris\Desktop\Git\uc\examples\ca\2015-03-01_reserves_3.json"

with open(data_file, 'r') as f:
    data = json.load(f)

thermal_gens = data["thermal_generators"]
demand = data["demand"]
num_periods = len(demand)

print("="*80)
print("АНАЛИЗ ПРОБЛЕМЫ НЕДОСТИЖИМОСТИ НА ПЕРИОДЕ 32")
print("="*80)

# Вычислить общую доступную мощность
total_capacity = sum(gen["power_output_maximum"] for gen in thermal_gens.values())

print(f"\nОбщая характеристика системы:")
print(f"  Количество генераторов: {len(thermal_gens)}")
print(f"  Общая установленная мощность: {total_capacity:.2f} MW")
print(f"  Количество периодов: {num_periods}")

# Анализ спроса по периодам
print(f"\n{'='*80}")
print("ДИНАМИКА СПРОСА (0-based индексы)")
print("="*80)

print(f"{'Период':<10} {'Спрос, MW':<15} {'% от макс мощности':<20}")
print("-"*80)

for t_idx in range(num_periods):
    period_demand = demand[t_idx]
    percent = (period_demand / total_capacity) * 100
    marker = " <-- КРИТИЧЕСКИЙ" if t_idx >= 30 else ""
    print(f"{t_idx:<10} {period_demand:<15.2f} {percent:<20.1f}%{marker}")

# Найти минимальный и максимальный спрос
min_demand = min(demand)
max_demand = max(demand)
avg_demand = sum(demand) / len(demand)

print(f"\n{'='*80}")
print("СТАТИСТИКА СПРОСА")
print("="*80)
print(f"  Минимальный спрос: {min_demand:.2f} MW")
print(f"  Максимальный спрос: {max_demand:.2f} MW")
print(f"  Средний спрос: {avg_demand:.2f} MW")
print(f"  Диапазон: {max_demand - min_demand:.2f} MW ({((max_demand - min_demand)/min_demand)*100:.1f}% от минимума)")

# Проверка ramping capability
print(f"\n{'='*80}")
print("АНАЛИЗ RAMPING CAPABILITY")
print("="*80)

# Подсчитать общий ramping up capability всех генераторов
total_ramp_up_per_period = sum(gen.get("ramp_up_limit", 0) for gen in thermal_gens.values())
total_ramp_down_per_period = sum(gen.get("ramp_down_limit", 0) for gen in thermal_gens.values())

print(f"  Общий ramp-up за период (все генераторы): {total_ramp_up_per_period:.2f} MW")
print(f"  Общий ramp-down за период (все генераторы): {total_ramp_down_per_period:.2f} MW")

# Проверить максимальное изменение спроса между соседними периодами
max_increase = 0
max_decrease = 0
critical_jump_period = None

for t_idx in range(1, num_periods):
    delta = demand[t_idx] - demand[t_idx - 1]
    if delta > max_increase:
        max_increase = delta
        critical_jump_period = t_idx
    if delta < max_decrease:
        max_decrease = delta

print(f"\n  Максимальное увеличение спроса между периодами: {max_increase:.2f} MW")
if critical_jump_period:
    print(f"    Произошло между периодами {critical_jump_period-1} и {critical_jump_period}")
    print(f"    ({demand[critical_jump_period-1]:.2f} -> {demand[critical_jump_period]:.2f} MW)")
print(f"  Максимальное снижение спроса между периодами: {abs(max_decrease):.2f} MW")

# Проверить может ли система справиться с таким скачком
if max_increase > total_ramp_up_per_period:
    print(f"\n  WARNING: КРИТИЧЕСКАЯ ПРОБЛЕМА:")
    print(f"  Максимальное увеличение спроса ({max_increase:.2f} MW) ПРЕВЫШАЕТ")
    print(f"  общий ramp-up capability ({total_ramp_up_per_period:.2f} MW)")
    print(f"  Дефицит ramp-up: {max_increase - total_ramp_up_per_period:.2f} MW")
    print(f"\n  Это означает, что НЕВОЗМОЖНО удовлетворить спрос в периоде {critical_jump_period}")
    print(f"  даже если ВСЕ генераторы работают на полную мощность!")
else:
    print(f"\n  OK: Теоретически система может справиться с максимальным скачком спроса")

# Анализ min_uptime/min_downtime ограничений
print(f"\n{'='*80}")
print("АНАЛИЗ MIN_UPTIME/MIN_DOWNTIME")
print("="*80)

uptime_distribution = {}
downtime_distribution = {}

for gen in thermal_gens.values():
    min_uptime = gen.get("time_up_minimum", 0)
    min_downtime = gen.get("time_down_minimum", 0)

    uptime_distribution[min_uptime] = uptime_distribution.get(min_uptime, 0) + 1
    downtime_distribution[min_downtime] = downtime_distribution.get(min_downtime, 0) + 1

print("Распределение min_uptime:")
for uptime, count in sorted(uptime_distribution.items(), reverse=True):
    print(f"  {uptime} периодов: {count} генераторов")

print("\nРаспределение min_downtime:")
for downtime, count in sorted(downtime_distribution.items(), reverse=True):
    print(f"  {downtime} периодов: {count} генераторов")

# Найти генераторы с большими ограничениями
max_uptime = max(gen.get("time_up_minimum", 0) for gen in thermal_gens.values())
max_downtime = max(gen.get("time_down_minimum", 0) for gen in thermal_gens.values())

print(f"\nМаксимальные ограничения:")
print(f"  Максимальный min_uptime: {max_uptime} периодов")
print(f"  Максимальный min_downtime: {max_downtime} периодов")

# Рекомендации
print(f"\n{'='*80}")
print("РЕКОМЕНДАЦИИ ДЛЯ RELAX-AND-FIX")
print("="*80)

if max_increase > total_ramp_up_per_period * 0.8:
    print("\n1. КРИТИЧНО: Увеличить window_size до минимум 12-16 периодов")
    print("   Текущий скачок спроса слишком большой для текущего окна")

if max_uptime > 8 or max_downtime > 8:
    print(f"\n2. ВАЖНО: Есть генераторы с min_uptime/downtime до {max(max_uptime, max_downtime)} периодов")
    print(f"   Установить window_step >= {max(max_uptime, max_downtime)} для избежания конфликтов")

print("\n3. РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ:")
print(f"   window_size = 16  (было 8)")
print(f"   window_step = 4   (было 8)")
print(f"   generators_per_iteration = 30-50  (было 61)")
print(f"   use_limited_horizon = False  (отключить адаптивный lookahead)")

print("\n4. Или попробовать классический Relax-and-Fix:")
print(f"   generators_per_iteration = None  (все генераторы сразу)")
print(f"   window_size = 12")
print(f"   window_step = 6")

print("\n" + "="*80)
