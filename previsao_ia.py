import ee
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 1. Autenticação
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# 2. Função de Extração (CORRIGIDA)
def extrair_dados_historicos(lat, lon):
    print(f"📡 Baixando dados históricos para Lat: {lat}, Lon: {lon}...")
    ponto = ee.Geometry.Point([lon, lat])
    
    # Pegando dados desde 2020 
    colecao = (ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
               .filterBounds(ponto)
               .filterDate('2020-01-01', '2025-01-01') 
               .select('NO2_column_number_density'))

    def extrair_valor(imagem):
        data = imagem.date().format("YYYY-MM-dd")
        # Redução da região
        dict_valor = imagem.reduceRegion(ee.Reducer.mean(), ponto, 1000)
        # Pegamos o valor com segurança no servidor
        val = dict_valor.get('NO2_column_number_density')
        return ee.Feature(None, {'ds': data, 'y': val})

    dados = colecao.map(extrair_valor).getInfo()
    
    lista = []
    for item in dados['features']:
        p = item['properties']
        
        # --- A CORREÇÃO ESTÁ AQUI ---
        # Usamos .get('y') para não dar erro se for None/Null
        valor_y = p.get('y')
        
        if valor_y is not None:
            lista.append([p['ds'], valor_y])
            
    df = pd.DataFrame(lista, columns=['ds', 'y'])
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    return df

# 3. Função de Previsão
def gerar_previsao(df_historico, anos_futuros=2):
    print("🔮 Treinando o modelo de IA (Prophet)...")
    
    # Ajuste de sensibilidade: daily_seasonality=False evita overfitting em ruídos
    modelo = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    
    modelo.fit(df_historico)
    
    futuro = modelo.make_future_dataframe(periods=365 * anos_futuros)
    previsao = modelo.predict(futuro)
    
    return modelo, previsao

# --- EXECUÇÃO ---

lat, lon = -23.5505, -46.6333

df = extrair_dados_historicos(lat, lon)

if not df.empty:
    print(f"✅ Histórico recuperado: {len(df)} pontos de dados.")
    
    # Prevendo 3 anos à frente
    modelo, forecast = gerar_previsao(df, anos_futuros=3) 
    
    print("✅ Previsão concluída! Gerando gráfico...")

    plt.figure(figsize=(14, 7))
    
    # Dados Reais
    plt.scatter(df['ds'], df['y'], color='black', s=5, label='Dados Reais')
    
    # Previsão
    plt.plot(forecast['ds'], forecast['yhat'], color='#0077b6', linewidth=2, label='Tendência (IA)')
    
    # Intervalo de Confiança (Sombra)
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='#0077b6', alpha=0.2)

    plt.title(f'Previsão de Poluentes (NO2) - IA Prophet\nLat: {lat}, Lon: {lon}', fontsize=16)
    plt.xlabel('Ano')
    plt.ylabel('Concentração NO2 (mol/m²)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    hoje = pd.Timestamp.now()
    plt.axvline(hoje, color='red', linestyle='--', label='Hoje')
    
    plt.tight_layout()
    plt.show()

else:
    print("Erro: Sem dados suficientes (verifique se a região não está muito nublada).")