import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os

class DataCollector:
    def __init__(self, exchange_name='binance'):
        """
        Инициализация коллектора данных
        """
        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
    def get_historical_data(self, symbol, timeframe='15m', limit=1000, since=None):
        """
        Получение исторических данных
        
        Args:
            symbol (str): Торговая пара (например, 'ADA/USDT')
            timeframe (str): Временной интервал ('1m', '5m', '15m', '1h', '4h', '1d')
            limit (int): Количество свечей
            since (int): Время начала в миллисекундах
            
        Returns:
            pd.DataFrame: DataFrame с историческими данными
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            # Если биржа возвращает кортеж (ohlcv, _), берем только первый элемент
            if isinstance(ohlcv, tuple) and len(ohlcv) == 2:
                ohlcv = ohlcv[0]
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Ошибка при получении данных: {e}")
            return None
    

    
    def get_data_for_period(self, symbol, timeframe='15m', years_back=None, days_back=None, start_date=None, end_date=None):
        """
        Универсальный метод получения данных за указанный период
        
        Args:
            symbol (str): Торговая пара
            timeframe (str): Временной интервал
            years_back (int): Количество лет назад от текущего момента
            days_back (int): Количество дней назад от текущего момента
            start_date (str): Дата начала в формате 'YYYY-MM-DD'
            end_date (str): Дата окончания в формате 'YYYY-MM-DD' (по умолчанию текущая дата)
            
        Returns:
            pd.DataFrame: DataFrame с данными за указанный период
        """
        print(f"Начинаем сбор данных для {symbol} с интервалом {timeframe}")
        
        # Определяем период
        end_time = datetime.now()
        
        if start_date:
            # Если указана дата начала
            start_time = datetime.strptime(start_date, '%Y-%m-%d')
            if end_date:
                end_time = datetime.strptime(end_date, '%Y-%m-%d')
        elif years_back:
            # Если указано количество лет назад
            start_time = end_time - timedelta(days=years_back * 365)
            print(f"Используем период: {years_back} лет назад")
        elif days_back:
            # Если указано количество дней назад
            start_time = end_time - timedelta(days=days_back)
        else:
            # По умолчанию 2 года назад
            start_time = end_time - timedelta(days=730)
            print("Не указан период, используем 2 года по умолчанию")
        
        print(f"Период: с {start_time.strftime('%Y-%m-%d %H:%M')} по {end_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Конвертируем в миллисекунды
        since = int(start_time.timestamp() * 1000)
        
        all_data = []
        current_since = since
        
        while current_since < int(end_time.timestamp() * 1000):
            print(f"Получаем данные с {datetime.fromtimestamp(current_since/1000)}")
            
            # Получаем порцию данных
            df_chunk = self.get_historical_data(symbol, timeframe, 1000, current_since)
            
            if df_chunk is None or df_chunk.empty:
                break
                
            all_data.append(df_chunk)
            
            # Обновляем время для следующей порции
            last_timestamp = df_chunk.index[-1]
            current_since = int(last_timestamp.timestamp() * 1000) + 1
            
            # Небольшая задержка чтобы не превысить лимиты API
            time.sleep(0.1)
        
        if all_data:
            # Объединяем все данные
            final_df = pd.concat(all_data, ignore_index=False)
            final_df = final_df[~final_df.index.duplicated(keep='first')]
            final_df.sort_index(inplace=True)
            
            print(f"Собрано {len(final_df)} свечей для {symbol}")
            print(f"Период: с {final_df.index[0]} по {final_df.index[-1]}")
            
            return final_df
        else:
            print("Не удалось получить данные")
            return None
    
    def get_recent_data(self, symbol, timeframe='15m', limit=1000):
        """
        Получение последних свечей
        
        Args:
            symbol (str): Торговая пара
            timeframe (str): Временной интервал
            limit (int): Количество последних свечей
            
        Returns:
            pd.DataFrame: DataFrame с последними свечами
        """
        print(f"Получаем последние {limit} свечей для {symbol} с интервалом {timeframe}")
        
        try:
            df = self.get_historical_data(symbol, timeframe, limit)
            if df is not None:
                print(f"Получено {len(df)} свечей")
                print(f"Период: с {df.index[0]} по {df.index[-1]}")
            return df
        except Exception as e:
            print(f"Ошибка при получении последних данных: {e}")
            return None
    
    def save_data(self, df, symbol, timeframe='15m', period_info=None):
        """
        Сохранение данных в CSV файл
        
        Args:
            df (pd.DataFrame): DataFrame с данными
            symbol (str): Торговая пара
            timeframe (str): Временной интервал
            period_info (str): Дополнительная информация о периоде (опционально)
        """
        if df is not None:
            # Создаем папку data если её нет
            os.makedirs('data', exist_ok=True)
            
            # Формируем имя файла
            symbol_clean = symbol.replace('/', '_')
            
            if period_info:
                filename = f"data/{symbol_clean}_{timeframe}_{period_info}.csv"
            else:
                filename = f"data/{symbol_clean}_{timeframe}.csv"
            
            # Сохраняем данные
            df.to_csv(filename)
            print(f"Данные сохранены в {filename}")
            
            return filename
        else:
            print("Нет данных для сохранения")
            return None
    
    def load_data(self, symbol, timeframe='15m', period_info=None):
        """
        Загрузка данных из CSV файла
        
        Args:
            symbol (str): Торговая пара
            timeframe (str): Временной интервал
            period_info (str): Дополнительная информация о периоде (опционально)
            
        Returns:
            pd.DataFrame: DataFrame с данными
        """
        symbol_clean = symbol.replace('/', '_')
        
        # Пробуем разные варианты названий файлов
        possible_filenames = []
        
        if period_info:
            possible_filenames.append(f"data/{symbol_clean}_{timeframe}_{period_info}.csv")
        
        possible_filenames.append(f"data/{symbol_clean}_{timeframe}.csv")
        
        # Для обратной совместимости
        possible_filenames.append(f"data/{symbol_clean}_{timeframe}_4years.csv")
        possible_filenames.append(f"data/{symbol_clean}_{timeframe}_2years.csv")
        
        for filename in possible_filenames:
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                print(f"Загружены данные из {filename}")
                print(f"Количество свечей: {len(df)}")
                print(f"Период: с {df.index[0]} по {df.index[-1]}")
                return df
        
        print(f"Файлы данных не найдены. Искали: {', '.join(possible_filenames)}")
        return None

if __name__ == "__main__":
    # Пример использования
    collector = DataCollector()
    
    symbol = "ADA/USDT"
    timeframe = "15m"
    
    # Примеры использования универсальной функции
    
    print("=== Пример 1: Последние 100 свечей ===")
    df_recent = collector.get_recent_data(symbol, timeframe, limit=100)
    
    print("\n=== Пример 2: Данные за последние 30 дней ===")
    df_30d = collector.get_data_for_period(symbol, timeframe, days_back=30)
    
    print("\n=== Пример 3: Данные за конкретный период ===")
    df_period = collector.get_data_for_period(symbol, timeframe, 
                                            start_date='2024-01-01', 
                                            end_date='2024-02-01')
    
    # Проверяем есть ли уже сохраненные данные
    df = collector.load_data(symbol, timeframe)
    
    if df is None:
        # Если данных нет, собираем их за 2 года
        print("\n=== Сбор данных за 2 года ===")
        df = collector.get_data_for_period(symbol, timeframe, days_back=730)
        if df is not None:
            collector.save_data(df, symbol, timeframe, period_info="2years")
    
    if df is not None:
        print("\nПервые 5 строк данных:")
        print(df.head())
        print("\nПоследние 5 строк данных:")
        print(df.tail())
        print(f"\nОбщая статистика:")
        print(df.describe())
