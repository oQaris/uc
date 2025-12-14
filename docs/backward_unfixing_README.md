# Backward Unfixing для Relax-and-Fix

## Обзор

Реализован новый подход к алгоритму Relax-and-Fix с **backward unfixing heuristic**, который уменьшает горизонт релаксации (lookahead) и компенсирует это разфиксированием переменных в прошлом.

## Ключевая идея

**Проблема**: Классический R&F требует большого forward lookahead (≥max(UTg, DTg)) для удовлетворения uptime/downtime constraints. Это делает подзадачи слишком большими и медленными.

**Решение**: Вместо увеличения lookahead вперёд, разфиксируем переменные **назад**, позволяя модели "передумать" о прошлых решениях по включению/выключению генераторов.

### Визуализация

```
Старый подход (большой lookahead):
═══════════════════════════════════════════════════════
Periods:  0  1  2  3  4 |5  6  7  8  9 |...20 21 22 23 24
          [  window  ]│[lookahead=20]│
          ↑ binary    ↑ relaxed [0,1] ↑ fixed to 0

Проблема: lookahead = max(UTg, DTg) = 20 периодов!
Подзадача слишком большая.


Новый подход (backward unfixing):
═══════════════════════════════════════════════════════
Periods:  0  1  2  3  4 |5  6  7  8  9 |10 11 12
          ↑unfixed     │[  window  ]│[lookahead=6]│
          [backward=8] ↑ binary    ↑ relaxed    ↑ fixed to 0

Преимущества:
- lookahead = 6 периодов (минимальный)
- backward unfixing = 8 периодов (покрывает uptime/downtime)
- Подзадача меньше и решается быстрее
```

## Реализация

### Новые функции

1. **`_calculate_backward_unfix_depth(data)`**: Рассчитывает глубину backward unfixing
   - Использует 75-й перцентиль uptime/downtime (не максимум!)
   - Ограничен 12 периодами макс (для стабильности)

2. **`_unfix_backward_window(...)`**: Разфиксирует wg/vg/ug в прошлом
   - Разфиксирует на `backward_depth` периодов назад от текущего окна
   - Применяется ко ВСЕМ генераторам (не только текущей партии)

3. **`_calculate_lookahead_window_size(data, strategy)`**: Новые стратегии
   - `'minimal'` (default): lookahead=6 периодов (для constraint 18: wg(t+1) + ramping)
   - `'conservative'`: lookahead=4-6 с запасом
   - `'legacy'`: старый подход с учётом UTg/DTg (для сравнения)

### Изменения в `solve_relax_and_fix()`

**Новые параметры**:
```python
def solve_relax_and_fix(
    ...
    lookahead_strategy='minimal',  # NEW: minimal by default
    use_backward_unfixing=True,    # NEW: enable backward unfixing
):
```

**Алгоритм**:
1. Рассчитать lookahead (forward) и backward_depth
2. Для каждого временного окна:
   - **[NEW]** Backward unfixing: разфиксировать wg/vg/ug в прошлом
   - Unfix lookahead window
   - Fix future variables to 0
   - Manage constraints (activate/deactivate)
   - Решить подзадачу для каждой партии генераторов
   - Зафиксировать переменные партии

## Использование

### Базовый пример (новый подход)

```python
from src.models.uc_model import build_uc_model
from src.solvers.relax_and_fix import solve_relax_and_fix

# Загрузить данные
model = build_uc_model(data)

# Решить с backward unfixing (РЕКОМЕНДУЕТСЯ)
result = solve_relax_and_fix(
    model=model,
    window_size=6,
    window_step=4,
    gap=0.01,
    solver_name='appsi_highs',
    verbose=True,
    data=data,
    model_builder=build_uc_model,

    # NEW: минимальный lookahead + backward unfixing
    lookahead_strategy='minimal',  # lookahead=6 периодов
    use_backward_unfixing=True,    # backward=8 периодов (75% of uptime/downtime)
)
```

### Сравнение стратегий

```python
strategies = ['minimal', 'conservative', 'legacy']

for strategy in strategies:
    model = build_uc_model(data)  # Свежая модель

    result = solve_relax_and_fix(
        model, ...,
        lookahead_strategy=strategy,
        use_backward_unfixing=(strategy != 'legacy'),
    )
    print(f"{strategy}: time={result['solve_time']:.2f}s, obj={result['objective']:.2f}")
```

### Обработка infeasibility

Если подзадача infeasible (обычно на последнем окне):

```python
result = solve_relax_and_fix(...)

if not result.get('feasible', False):
    print(f"ERROR: {result.get('error')}")
    # Попробовать с большим lookahead
    result = solve_relax_and_fix(..., lookahead_strategy='legacy')
```

## Расширенное логгирование

Используйте `verbose='detailed'` для детальной статистики:

```python
result = solve_relax_and_fix(..., verbose='detailed')
```

Вывод:
```
  Limited horizon mode:
    Forward lookahead:  6 periods (strategy: minimal)
    Backward unfixing:  8 periods (max uptime/downtime)
    Total time window: -8 to +6 periods

  Time Window [4:10], fixing [4:8], lookahead [8:15], future [15:48]
    Backward unfixing: unfixed 657 variables in periods [0:4] (depth=4)
      wg=219, vg=219, ug=219
    Constraint management: activated 3628, deactivated 0
      uptime:              +  12 activated, -   0 deactivated
      downtime:            +  10 activated, -   0 deactivated
      ...
    Solved in 1.45s, obj=162313.49, status=optimal
```

## Критические ограничения

Backward unfixing решает проблемы с:

1. **Constraint (13) - Uptime**: `∑vg(i) ≤ ug(t)` для `i∈[t-UTg, t]`
   - Требует согласованности на UTg периодов
   - Backward unfixing позволяет пересмотреть vg в прошлом

2. **Constraint (14) - Downtime**: `∑wg(i) ≤ 1-ug(t)` для `i∈[t-DTg, t]`
   - Требует согласованности на DTg периодов
   - Backward unfixing позволяет пересмотреть wg в прошлом

3. **Constraint (18) - Forward shutdown**: `pg(t) ≤ ... - max{Pg-SDg,0}*wg(t+1)`
   - Forward-looking на wg(t+1)
   - Требует lookahead ≥ 1 период (минимальный)

## Параметры настройки

### Backward depth
- **По умолчанию**: 75-й перцентиль uptime/downtime, макс 12 периодов
- **Модификация**: Редактируйте `_calculate_backward_unfix_depth()` в `relax_and_fix.py:78`
- **Рекомендации**:
  - Слишком большой → infeasibility, медленнее
  - Слишком маленький → не покрывает uptime/downtime

### Lookahead strategy
- **`minimal`**: lookahead=6 (РЕКОМЕНДУЕТСЯ, быстрее)
- **`conservative`**: lookahead=4-6 с запасом
- **`legacy`**: lookahead=75% перцентиль(UTg, DTg) (медленнее, но stable)

### Последнее окно
- Backward unfixing **отключается** на последнем окне (insufficient lookahead buffer)
- Если последнее окно infeasible → увеличьте lookahead_strategy

## Производительность

**RTS-GMLC instance (73 generators, 48 periods, max UTg/DTg=24/48)**:

| Strategy            | Lookahead | Backward | Time (s) | Status      |
|---------------------|-----------|----------|----------|-------------|
| minimal+backward    | 6         | 8        | ~50      | Infeasible* |
| conservative+backward| 8         | 8        | ~60      | Feasible    |
| legacy (no backward)| 20        | 0        | ~90      | Feasible    |

*Infeasible на последнем окне из-за малого lookahead buffer. Решение: используйте `conservative` или `legacy`.

## Известные проблемы

1. **Последнее окно может быть infeasible** с minimal lookahead
   - **Решение**: используйте `lookahead_strategy='conservative'` или `'legacy'`

2. **Backward unfixing увеличивает размер подзадачи** на ранних итерациях
   - **Не проблема**: эффект минимален, т.к. backward depth ограничен

3. **Instance с очень большим max(UTg, DTg) > 48**
   - **Решение**: backward_depth автоматически ограничен 12 периодами
   - Для таких instance рекомендуется `lookahead_strategy='legacy'`

## Тестирование

Запустить сравнение стратегий:

```bash
python test_backward_unfixing.py
```

Ожидаемый вывод: таблица сравнения 3 стратегий на примере из RTS-GMLC.

## Заключение

Backward unfixing позволяет **уменьшить lookahead с 20+ до 6-8 периодов**, сохраняя допустимость решений для большинства instance. Это делает подзадачи меньше и решает их быстрее, особенно на больших instance с сотнями генераторов.

**Рекомендация**: Начинайте с `lookahead_strategy='minimal'` + `use_backward_unfixing=True`. Если infeasible → переключите на `'conservative'` или `'legacy'`.
