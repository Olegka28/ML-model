# 🚀 Модульная ML система для торговли криптовалютами

Современная, масштабируемая система машинного обучения для прогнозирования движения цен криптовалют с использованием технического анализа и multi-timeframe признаков.

## 🎯 Возможности системы

### 📊 Анализ данных
- **Multi-timeframe анализ**: 15m, 1h, 4h, 1d таймфреймы
- **141 технический индикатор**: RSI, MACD, Bollinger Bands, ATR, EMA, ADX, и многие другие
- **Lag и rolling признаки**: временные паттерны и тренды
- **Автоматическая валидация данных**: проверка качества OHLCV данных
- **Гибкий сбор данных**: поддержка различных периодов и источников

### 🤖 Алгоритмы машинного обучения
- **XGBoost** - градиентный бустинг с оптимизацией гиперпараметров
- **LightGBM** - быстрый градиентный бустинг
- **CatBoost** - категориальный бустинг
- **Автоматическая оптимизация** через Optuna (до 1000 trials)
- **Time Series Cross-Validation**: Walk-Forward, TimeSeriesSplit, Expanding Window

### 🎯 Типы задач
- **Регрессия**: прогнозирование процентного изменения цены
  - `crypto_clipped` - крипто-оптимизированное обрезание (рекомендуется)
  - `volume_weighted` - взвешенная по объему доходность
  - `vol_regime` - адаптивная к волатильности
  - `market_regime` - адаптивная к режиму рынка
  - `momentum_enhanced` - улучшенный момент
  - `volume_volatility` - объемно-волатильная доходность
- **Классификация**: прогнозирование направления движения
  - Бинарная классификация с настраиваемым процентом роста
  - Автоматическая балансировка классов (SMOTE)

### 📈 Горизонты прогнозирования
- **Краткосрочные**: 3-10 баров (внутридневная торговля)
- **Среднесрочные**: 15-30 баров (свинг-трейдинг)
- **Долгосрочные**: 50+ баров (позиционная торговля)

## 🏗️ Архитектура системы

```
ml_module/
├── core/                    # Основные компоненты
│   ├── base_system.py      # Базовый координатор (исправлен)
│   ├── model_manager.py    # Обучение моделей (улучшен)
│   ├── model_versioning.py # Версификация моделей (новый)
│   └── baseline.py         # Baseline модели (новый)
├── data_collector/          # Сбор и управление данными
│   ├── data_collector.py   # Скачивание с бирж (улучшен)
│   ├── data_manager.py     # Управление данными (перемещен)
│   └── __init__.py         # Пакет data_collector
├── features/               # Управление признаками (реорганизован)
│   ├── feature_manager.py  # Генерация и кэширование признаков
│   ├── feature_engineer.py # Создание технических индикаторов
│   ├── target_creator.py   # Создание целевых переменных (улучшен)
│   └── __init__.py         # Пакет features
├── systems/                # Специализированные системы (улучшены)
│   ├── regression_system.py    # Система регрессии
│   └── classification_system.py # Система классификации
├── utils/                  # Утилиты
│   ├── config.py          # Конфигурация (обновлена)
│   ├── logger.py          # Логирование
│   └── validators.py      # Валидация
└── utils/                  # Утилиты
```

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Подготовка данных
Разместите CSV файлы в папке `data/` в формате:
```
data/
├── SOL_USDT_15m.csv        # Основной формат (без префиксов)
├── SOL_USDT_1h.csv
├── SOL_USDT_4h.csv
├── SOL_USDT_1d.csv
# Или с дополнительной информацией (для обратной совместимости):
├── SOL_USDT_15m_2years.csv
├── SOL_USDT_1h_2years.csv
├── SOL_USDT_4h_2years.csv
└── SOL_USDT_1d_2years.csv
```

### 3. Обучение модели
```bash
# Базовая регрессионная модель (использует RegressionSystem)
python train_model.py train --symbol SOL_USDT --target crypto_clipped --trials 100

# Классификационная модель (использует ClassificationSystem)
python train_model.py train --symbol SOL_USDT --task classification --percent 0.025 --trials 100

# Просмотр доступных моделей
python train_model.py list

# Информация о модели
python train_model.py info --symbol SOL_USDT --task regression
```

### 4. Получение предсказаний
```bash
# Предсказание регрессионной модели (использует RegressionSystem)
python predict.py SOL_USDT --task regression

# Предсказание классификационной модели (использует ClassificationSystem)
python predict.py SOL_USDT --task classification

# Сравнение обеих моделей
python predict.py SOL_USDT --task compare

# Информация о модели
python predict.py SOL_USDT --task info
```

## 📊 Примеры использования

### Внутридневная торговля (15m)
```bash
python train_model.py train \
  --symbol SOL_USDT \
  --target crypto_clipped \
  --horizon 5 \
  --timeframes 15m 1h \
  --trials 200
```

### Свинг-трейдинг (4h)
```bash
python train_model.py train \
  --symbol SOL_USDT \
  --target volume_weighted \
  --horizon 20 \
  --timeframes 15m 1h 4h 1d \
  --trials 300
```

### Классификация направления (2.5% рост)
```bash
python train_model.py train \
  --symbol SOL_USDT \
  --task classification \
  --percent 0.025 \
  --horizon 10 \
  --trials 150
```

### Классификация направления (5% рост)
```bash
python train_model.py train \
  --symbol SOL_USDT \
  --task classification \
  --percent 0.05 \
  --horizon 15 \
  --trials 200
```

## 📈 Интерпретация результатов

### Регрессионные модели
- **Положительное значение** → ожидается рост цены
- **Отрицательное значение** → ожидается падение цены
- **Значение близкое к 0** → ожидается боковое движение
- **Уверенность** → насколько модель уверена в предсказании

### Классификационные модели
- **Класс 1** → ожидается рост (РОСТ)
- **Класс 0** → ожидается падение (ПАДЕНИЕ)
- **Уверенность** → насколько модель уверена в предсказании
- **Процент роста** → настраиваемый порог для классификации

### Метрики качества
- **RMSE** - среднеквадратичная ошибка (чем меньше, тем лучше)
- **MAE** - средняя абсолютная ошибка
- **R²** - коэффициент детерминации (ближе к 1 = лучше)
- **Accuracy** - точность классификации
- **F1-score** - гармоническое среднее precision и recall
- **Precision** - точность положительных предсказаний
- **Recall** - полнота положительных предсказаний

## 🔧 Конфигурация системы

### Основные параметры
```python
from ml_module.utils.config import Config

config = Config(
    models_root='models',           # Папка для сохранения моделей
    data_root='data',               # Папка с данными
    cache_root='cache',             # Папка для кэша
    log_level='INFO',               # Уровень логирования
    mlflow_tracking_uri='sqlite:///mlruns.db',  # MLflow
    mlflow_experiment_name='crypto_trading',    # Эксперимент
    # Новые параметры для Time Series CV
    use_time_series_cv=True,        # Использовать Time Series CV
    cv_type='walk_forward',         # Тип CV: walk_forward/time_series/expanding
    cv_n_splits=5,                  # Количество сплитов
    cv_test_size=0.2                # Размер тестового набора
)
```

### Параметры модели
```python
model_config = {
    'model_type': 'xgboost',        # xgboost/lightgbm/catboost
    'n_trials': 100,                # Количество Optuna trials
    'early_stopping_rounds': 20,    # Ранняя остановка
    'target_type': 'crypto_clipped',       # Тип таргета (обновлен)
    'horizon': 10                   # Горизонт прогнозирования
}
```

## 🆕 Новые возможности

### 🎯 Улучшенные системы
- **RegressionSystem**: специализированная система для регрессии
- **ClassificationSystem**: специализированная система для классификации
- **Автоматическая балансировка классов** с SMOTE
- **Гибкие параметры классификации** (настраиваемый процент роста)

### 📊 Улучшенное управление данными
- **Гибкий сбор данных**: поддержка различных периодов
- **Улучшенное кэширование**: in-memory и disk-based кэш
- **Валидация данных**: проверка качества и консистентности
- **Обратная совместимость**: поддержка старых форматов файлов

### 🤖 Улучшенное обучение моделей
- **Time Series Cross-Validation**: Walk-Forward, TimeSeriesSplit, Expanding Window
- **Baseline модели**: Dummy, Linear Regression, Random Forest, Persistence
- **Автоматическое сравнение** с baseline моделями
- **Улучшенная версификация** с валидацией и очисткой

### 🔍 Улучшенная аналитика
- **Анализ важности признаков** с читаемыми именами
- **Стабильность признаков** через множественные методы
- **Корреляционный анализ** с автоматическим удалением
- **Mutual Information** для нелинейных зависимостей

### 💾 Улучшенное сохранение
- **Умная версификация** с автоматическим сравнением
- **Валидация моделей** перед сохранением
- **Автоматическая очистка** старых версий
- **Экспорт моделей** в различных форматах (pickle, JSON, ONNX)

## 📊 Анализ важности признаков

Система автоматически анализирует важность признаков с читаемыми именами:

```bash
# Результаты сохраняются в:
optimization_results/SOL_USDT_xgboost_feature_importance.csv
```

Топ-10 наиболее важных признаков для SOL_USDT:
1. **rsi_14** - Relative Strength Index
2. **macd_12_26** - MACD индикатор
3. **bb_upper_20** - Верхняя полоса Bollinger Bands
4. **atr_14** - Average True Range
5. **ema_20_slope** - Наклон EMA 20
6. **volume_sma_20** - Объем относительно SMA
7. **stoch_k_14** - Stochastic %K
8. **adx_14** - Average Directional Index (исправлен)
9. **cci_20** - Commodity Channel Index
10. **williams_r_14** - Williams %R

## 🔍 Мониторинг и логирование

### MLflow эксперименты
Все эксперименты автоматически логируются в MLflow:
- Параметры моделей
- Метрики валидации
- Baseline сравнения
- Time Series CV результаты
- Время выполнения
- Лучшие гиперпараметры

### Файловые логи
```bash
# Просмотр логов
tail -f logs/ml_system_$(date +%Y%m%d).log
```

### Кэширование
Система автоматически кэширует:
- Загруженные данные
- Сгенерированные признаки
- Обученные модели
- Результаты валидации

## 🎯 Текущие возможности

### ✅ Реализовано
- [x] Multi-timeframe анализ данных
- [x] 141 технический индикатор (включая исправленный ADX)
- [x] Автоматическая оптимизация гиперпараметров
- [x] Три алгоритма ML (XGBoost, LightGBM, CatBoost)
- [x] Регрессия и классификация
- [x] Валидация данных и моделей
- [x] Улучшенная версификация моделей
- [x] MLflow интеграция
- [x] CLI интерфейсы (исправлены)
- [x] Анализ важности признаков с читаемыми именами
- [x] Baseline сравнения
- [x] Кэширование данных
- [x] Time Series Cross-Validation
- [x] Автоматическая балансировка классов (SMOTE)
- [x] Улучшенные системы (RegressionSystem, ClassificationSystem)
- [x] Гибкий сбор данных
- [x] Улучшенная обработка ошибок
- [x] Валидация входных данных

### 🔄 В разработке
- [ ] Ensemble модели (голосование/стэкинг)
- [ ] Временные ряды (LSTM, Transformer)
- [ ] Автоматический backtesting
- [ ] Risk management модуль
- [ ] Real-time предсказания
- [ ] Web интерфейс

## 🚀 Планы улучшений

### 1. Расширенные алгоритмы
- **Deep Learning**: LSTM, GRU, Transformer для временных рядов
- **Ensemble методы**: Stacking, Blending, Voting
- **Unsupervised Learning**: Clustering для сегментации рынка
- **Reinforcement Learning**: Q-learning для торговых стратегий

### 2. Расширенные признаки
- **On-chain данные**: транзакции, адреса, депозиты/выводы
- **Sentiment анализ**: новости, социальные сети, Reddit
- **Макроэкономические данные**: индекс страха, доминирование BTC
- **Корреляции**: с другими активами, индексами

### 3. Risk Management
- **Position Sizing**: автоматический расчет размера позиции
- **Stop Loss/Take Profit**: динамические уровни
- **Portfolio Optimization**: распределение между активами
- **Drawdown Protection**: защита от просадок

### 4. Автоматизация
- **Auto-retraining**: автоматическое переобучение моделей
- **Model Drift Detection**: обнаружение дрейфа модели
- **A/B Testing**: тестирование новых стратегий
- **Performance Monitoring**: мониторинг производительности

### 5. Интерфейсы
- **Web Dashboard**: веб-интерфейс для мониторинга
- **API**: REST API для интеграции
- **Telegram Bot**: уведомления и управление
- **Trading Integration**: прямая интеграция с биржами

### 6. Расширенная аналитика
- **Market Regime Detection**: определение режима рынка
- **Volatility Forecasting**: прогнозирование волатильности
- **Correlation Analysis**: анализ корреляций
- **Feature Engineering**: автоматическая генерация признаков

### 7. Производительность
- **GPU Acceleration**: использование GPU для обучения
- **Distributed Training**: распределенное обучение
- **Model Compression**: сжатие моделей для быстрого инференса
- **Caching Optimization**: оптимизация кэширования

### 8. Безопасность и надежность
- **Model Validation**: расширенная валидация моделей
- **Data Quality**: мониторинг качества данных
- **Backup & Recovery**: резервное копирование
- **Error Handling**: улучшенная обработка ошибок

## 📚 Документация

### Основные файлы
- `PRICE_GROWTH_MODEL_README.md` - документация по моделям роста цены
- `MULTICLASS_MODEL_README.md` - документация по многоклассовым моделям
- `TRADING_SIGNALS_README.md` - документация по торговым сигналам

### Примеры кода
```python
# Базовое использование с новой архитектурой
from ml_module.systems.regression_system import RegressionSystem
from ml_module.systems.classification_system import ClassificationSystem
from ml_module.utils.config import Config

# Конфигурация
config = Config(models_root='models', data_root='data')

# Регрессионная система
regression_system = RegressionSystem(config)
metadata = regression_system.run_experiment(
    symbol='SOL_USDT',
    target_type='crypto_clipped',
    horizon=10
)

# Классификационная система
classification_system = ClassificationSystem(config)
metadata = classification_system.run_experiment(
    symbol='SOL_USDT',
    percent=0.025,  # 2.5% рост
    horizon=20
)

# Получение предсказаний
from ml_module.data_collector import DataCollector
collector = DataCollector()
latest_data = collector.get_recent_data('SOL_USDT', '15m', 100)

# Предсказание регрессии
reg_prediction = regression_system.predict_latest('SOL_USDT', latest_data, '15m')

# Предсказание классификации
cls_prediction = classification_system.predict_latest('SOL_USDT', latest_data, '15m')

# Информация о моделях
reg_info = regression_system.get_model_info('SOL_USDT')
cls_info = classification_system.get_model_info('SOL_USDT')
```

## 🤝 Вклад в проект

Приветствуются вклады в виде:
- Новых алгоритмов ML
- Дополнительных технических индикаторов
- Улучшений в документации
- Исправлений багов
- Оптимизации производительности

## 📄 Лицензия

MIT License - свободное использование для коммерческих и некоммерческих целей.

## ⚠️ Дисклеймер

Эта система предназначена для образовательных и исследовательских целей. Торговля криптовалютами связана с высокими рисками. Авторы не несут ответственности за финансовые потери.

---

**🎉 Система полностью обновлена и готова к использованию! Все критические проблемы исправлены, архитектура улучшена, добавлены новые возможности.**

### 🚀 Быстрый старт:
```bash
# Обучение регрессионной модели
python train_model.py train --symbol SOL_USDT --target crypto_clipped

# Обучение классификационной модели  
python train_model.py train --symbol SOL_USDT --task classification --percent 0.025

# Получение предсказаний
python predict.py SOL_USDT --task regression
python predict.py SOL_USDT --task classification

# Просмотр моделей
python train_model.py list
python predict.py SOL_USDT --task info
``` 