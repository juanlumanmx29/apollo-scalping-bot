import os
import pandas as pd
from binance.client import Client
from binance.enums import *

# --- Configuración de la API de Binance ---
# Para propósitos de prueba y desarrollo, puedes usar claves simuladas.
# Para producción, asegúrate de usar tus propias claves de API de Binance.
# NUNCA compartas tus claves API y guárdalas de forma segura.

# Puedes obtener tus claves API de las variables de entorno
# o reemplazarlas directamente aquí (NO RECOMENDADO PARA PRODUCCIÓN).
API_KEY = os.environ.get("BINANCE_API_KEY", "TU_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "TU_API_SECRET")

client = Client(API_KEY, API_SECRET)

def get_historical_klines(symbol, interval, start_str, end_str=None):
    """
    Obtiene datos históricos de velas (klines) de Binance.

    Args:
        symbol (str): El par de trading (ej. 'ETHUSDT').
        interval (str): El intervalo de tiempo de las velas (ej. Client.KLINE_INTERVAL_1MINUTE).
        start_str (str): Fecha de inicio en formato string (ej. '1 Jan, 2023').
        end_str (str, optional): Fecha de fin en formato string. Si es None, hasta ahora.

    Returns:
        pandas.DataFrame: DataFrame con los datos históricos de velas.
    """
    print(f"Obteniendo datos históricos para {symbol} con intervalo {interval} desde {start_str}...")
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)

    # Convertir a DataFrame de pandas
    df = pd.DataFrame(klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])

    # Convertir timestamps a datetime
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

    # Convertir columnas numéricas a tipo float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                    'Taker buy base asset volume', 'Taker buy quote asset volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df

if __name__ == "__main__":
    SYMBOL = 'ETHUSDT'
    INTERVALS = [
        Client.KLINE_INTERVAL_1MINUTE,
        Client.KLINE_INTERVAL_5MINUTE,
        Client.KLINE_INTERVAL_15MINUTE,
        Client.KLINE_INTERVAL_30MINUTE,
        Client.KLINE_INTERVAL_1HOUR,
        Client.KLINE_INTERVAL_4HOUR,
        Client.KLINE_INTERVAL_1DAY
    ]
    START_DATE = '1 Jan, 2024' # Puedes ajustar la fecha de inicio

    # Crear un directorio para guardar los datos si no existe
    output_dir = 'historical_data'
    os.makedirs(output_dir, exist_ok=True)

    for interval in INTERVALS:
        try:
            df_klines = get_historical_klines(SYMBOL, interval, START_DATE)
            if not df_klines.empty:
                file_name = f"{output_dir}/{SYMBOL}_{interval}.csv"
                df_klines.to_csv(file_name, index=False)
                print(f"Datos guardados en {file_name}")
            else:
                print(f"No se obtuvieron datos para {SYMBOL} con intervalo {interval}.")
        except Exception as e:
            print(f"Error al obtener datos para {SYMBOL} con intervalo {interval}: {e}")

    print("Recolección de datos históricos completada.")


