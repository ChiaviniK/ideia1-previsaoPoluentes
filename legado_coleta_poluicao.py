import requests
import pandas as pd
from datetime import datetime
import time

# --- CONFIGURAÇÃO ---
API_KEY = "8830e70422a578a38464b22e53b7ec1d"  # Cole sua chave aqui
LAT = "-23.5505"  # Exemplo: São Paulo
LON = "-46.6333"

# Definir intervalo de datas para o histórico (formato UNIX Timestamp)
# Exemplo: Coletar dados de 01/01/2023 até 01/01/2024
start_date = "01/01/2023"
end_date = "01/01/2024"

def get_unix_timestamp(date_str):
    """Converte string 'dd/mm/yyyy' para timestamp Unix."""
    dt_obj = datetime.strptime(date_str, "%d/%m/%Y")
    return int(time.mktime(dt_obj.timetuple()))

def fetch_pollution_history(lat, lon, start, end, api_key):
    """
    Busca histórico de poluição na API OpenWeatherMap.
    Retorna um DataFrame do Pandas limpo.
    """
    # Converter datas
    start_unix = get_unix_timestamp(start)
    end_unix = get_unix_timestamp(end)
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={api_key}"
    
    print(f"🔄 Consultando API para o período: {start} até {end}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Levanta erro se a requisição falhar
        data = response.json()
        
        # Verificar se existem dados na lista
        if 'list' not in data or len(data['list']) == 0:
            print("⚠️ Nenhum dado encontrado para este período/local.")
            return None
        
        # --- PROCESSAMENTO DOS DADOS (ETL) ---
        records = []
        for item in data['list']:
            # O timestamp vem em segundos, convertemos para data legível
            dt_readable = datetime.fromtimestamp(item['dt'])
            
            # Extraindo componentes químicos e AQI
            record = {
                'data_hora': dt_readable,
                'aqi': item['main']['aqi'], # Índice de Qualidade do Ar (1 = Bom, 5 = Péssimo)
                'co': item['components']['co'],     # Monóxido de Carbono
                'no': item['components']['no'],     # Monóxido de Nitrogênio
                'no2': item['components']['no2'],   # Dióxido de Nitrogênio
                'o3': item['components']['o3'],     # Ozônio
                'so2': item['components']['so2'],   # Dióxido de Enxofre
                'pm2_5': item['components']['pm2_5'], # Partículas Finas (importante!)
                'pm10': item['components']['pm10'],   # Partículas Inaláveis
                'nh3': item['components']['nh3']      # Amônia
            }
            records.append(record)
            
        # Criar DataFrame
        df = pd.DataFrame(records)
        print(f"✅ Sucesso! {len(df)} registros coletados.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Erro na conexão: {e}")
        return None

# --- EXECUÇÃO ---
if __name__ == "__main__":
    df_poluicao = fetch_pollution_history(LAT, LON, start_date, end_date, API_KEY)
    
    if df_poluicao is not None:
        # Visualizar as primeiras linhas
        print("\n--- Amostra dos Dados ---")
        print(df_poluicao.head())
        
        # Salvar em CSV para usar no próximo passo (Análise/Previsão)
        arquivo_saida = "historico_poluicao.csv"
        df_poluicao.to_csv(arquivo_saida, index=False)
        print(f"\n💾 Dados salvos em '{arquivo_saida}'")