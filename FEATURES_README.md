# Новые возможности скрипта run_all_tests_parallel.py

## ✨ Реализованные функции

### 1. 🔄 Автоматическое продолжение с места остановки (Resume)

Скрипт автоматически **пропускает уже решённые примеры** при повторном запуске с тем же output файлом.

**Как это работает:**
- При запуске скрипт читает существующий CSV файл
- Извлекает список примеров со статусом `optimal` или `feasible`
- Решает только те примеры, которых нет в списке

**Пример:**
```bash
# Первый запуск - решает все 56 примеров
python run_all_tests_parallel.py --output results.csv

# Скрипт прервался после 20 примеров (Ctrl+C или ошибка)

# Повторный запуск - автоматически продолжит с 21-го примера
python run_all_tests_parallel.py --output results.csv
```

**Вывод:**
```
Found 56 test instances
Found 20 already solved instances in results.csv
Skipping 20 already solved instances
Remaining to solve: 36
```

### 2. 📊 Метаданные о сложности примера

Каждый пример теперь содержит **подробную информацию** о своей структуре и сложности.

**Добавленные поля в CSV:**

| Поле | Описание |
|------|----------|
| `time_periods` | Количество временных периодов (обычно 48) |
| `n_thermal_gens` | Количество тепловых генераторов |
| `n_renewable_gens` | Количество возобновляемых источников |
| `n_must_run` | Количество генераторов с обязательной работой |
| `total_startup_categories` | Общее число категорий запуска |
| `total_pwl_points` | Общее число точек кусочно-линейной аппроксимации |
| `approx_binary_vars` | Приблизительное число бинарных переменных |
| `approx_continuous_vars` | Приблизительное число непрерывных переменных |
| `approx_total_vars` | Общее число переменных |
| `approx_constraints` | Приблизительное число ограничений |
| `peak_demand` | Пиковый спрос |
| `avg_demand` | Средний спрос |
| `total_reserves` | Общий объём резервов |

**Зачем это нужно:**
- Понять, почему одни примеры решаются дольше других
- Прогнозировать время решения новых примеров
- Анализировать зависимость времени от характеристик

**Пример анализа:**
```python
import pandas as pd

# Загрузить результаты
df = pd.read_csv('results.csv')

# Корреляция между сложностью и временем решения
correlation = df['solve_time'].corr(df['approx_binary_vars'])
print(f"Correlation with binary vars: {correlation:.3f}")

# Самые сложные примеры
hardest = df.nlargest(5, 'solve_time')[['instance', 'solve_time', 'approx_binary_vars']]
print(hardest)
```

## 📈 Анализ сложности примеров

### Что влияет на время решения:

1. **Бинарные переменные** (approx_binary_vars) - **основной фактор**
   - Больше бинарных переменных = дольше решение
   - Обычно ~146,400 переменных

2. **Категории запуска** (total_startup_categories)
   - Усложняют логику переключения генераторов
   - Обычно ~1,220 категорий

3. **Резервы** (total_reserves)
   - Примеры с резервами обычно сложнее
   - `reserves_0` = нет резервов (легче)
   - `reserves_5` = 5% резервов (сложнее)

4. **Must-run генераторы** (n_must_run)
   - Сокращают пространство поиска
   - Могут как ускорить, так и замедлить решение

### Наблюдения из test_results_parallel.csv:

```
Самые быстрые:
- 2015-06-01_reserves_0.json: 530s, reserves=0
- 2014-09-01_reserves_0.json: 569s, reserves=0

Самые медленные:
- 2015-03-01_reserves_5.json: 1572s, reserves=69546
- 2014-12-01_reserves_5.json: 1713s, reserves=?

Вывод: Примеры с reserves=5 (5% резервов) решаются ~3x дольше
```

## 🚀 Использование

### Базовый запуск с resume
```bash
python run_all_tests_parallel.py --parallel 16 --output my_results.csv
```

Если скрипт прервётся, просто запустите снова:
```bash
python run_all_tests_parallel.py --parallel 16 --output my_results.csv
```

### Принудительное перерешение всех примеров
```bash
# Используйте другое имя файла или удалите старый
rm my_results.csv
python run_all_tests_parallel.py --parallel 16 --output my_results.csv
```

### Решение только конкретных примеров
```bash
# Создайте новый CSV файл для этих примеров
python run_all_tests_parallel.py \
    --instances ca/Scenario400*.json \
    --output scenario400_results.csv
```

## 📊 Анализ результатов

### Загрузка и базовая статистика
```python
import pandas as pd
import matplotlib.pyplot as plt

# Загрузить результаты
df = pd.read_csv('results.csv')

# Базовая статистика
print(df[['solve_time', 'approx_binary_vars', 'total_reserves']].describe())

# График зависимости времени от резервов
plt.scatter(df['total_reserves'], df['solve_time'])
plt.xlabel('Total Reserves')
plt.ylabel('Solve Time (s)')
plt.title('Solve Time vs Reserves')
plt.show()
```

### Поиск выбросов
```python
# Примеры, которые решались аномально долго
mean_time = df['solve_time'].mean()
std_time = df['solve_time'].std()

outliers = df[df['solve_time'] > mean_time + 2*std_time]
print("Outliers:")
print(outliers[['instance', 'solve_time', 'total_reserves', 'n_must_run']])
```

### Группировка по типу резервов
```python
# Извлечь уровень резервов из имени файла
df['reserve_level'] = df['instance'].str.extract(r'reserves_(\d+)')

# Средние времена по уровням резервов
by_reserves = df.groupby('reserve_level')['solve_time'].agg(['mean', 'std', 'count'])
print(by_reserves)
```

## 🔍 Отладка и мониторинг

### Проверка прогресса во время выполнения
```bash
# В отдельном терминале
watch -n 5 'wc -l results.csv'

# Или
tail -f results.csv
```

### Проверка статуса решённых примеров
```python
import csv

solved = set()
with open('results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['status'] in ['optimal', 'feasible']:
            solved.add(row['instance'])

print(f"Solved {len(solved)} instances")
print(f"Examples: {list(solved)[:5]}")
```

## 💡 Советы

1. **Используйте resume для длительных запусков**
   - Не нужно бояться остановить скрипт
   - Можно запускать по частям

2. **Анализируйте метаданные перед полным запуском**
   - Запустите на 5-10 примерах
   - Оцените время на основе `approx_binary_vars`
   - Спрогнозируйте общее время

3. **Сортируйте примеры по сложности**
   - Начните с простых (reserves_0)
   - Затем переходите к сложным (reserves_5)

4. **Мониторьте прогресс**
   - Используйте `watch` или `tail -f`
   - Проверяйте загрузку CPU в диспетчере задач

## 🎯 Пример рабочего процесса

```bash
# 1. Тестовый запуск на 5 примерах
python run_all_tests_parallel.py --limit 5 --parallel 2 --output test.csv

# 2. Анализ результатов теста
python -c "import pandas as pd; df = pd.read_csv('test.csv'); print(df['solve_time'].mean())"

# 3. Полный запуск с resume
python run_all_tests_parallel.py --parallel 16 --output full_results.csv

# 4. Если прервался - просто перезапустить
python run_all_tests_parallel.py --parallel 16 --output full_results.csv

# 5. Финальный анализ
python analyze_results.py full_results.csv
```

## 📝 Структура CSV файла

```csv
instance,status,solve_time,build_time,load_time,total_time,objective_value,gap,error,file_path,time_periods,n_thermal_gens,n_renewable_gens,n_must_run,total_startup_categories,total_pwl_points,approx_binary_vars,approx_continuous_vars,approx_total_vars,approx_constraints,peak_demand,avg_demand,total_reserves
2014-09-01_reserves_0.json,optimal,122.18,9.92,0.0,132.09,48256.06,0.078,,.\ca\...,48,610,0,200,1220,1488,146400,159264,305664,355726,36856.37,28977.56,0.0
```

**22 колонки:**
- 9 колонок о результатах решения
- 13 колонок метаданных о примере

## ⚠️ Важные замечания

1. **Resume работает по имени файла**
   - Если переименовали файлы примеров - будут решены заново
   - Используйте один и тот же output файл для resume

2. **Статусы для resume**
   - Пропускаются только `optimal` и `feasible`
   - Примеры с ошибками будут перерешены

3. **Метаданные вычисляются до решения**
   - Даже если решение провалилось, метаданные будут записаны
   - Позволяет анализировать, почему пример провалился
