import ee
import pandas as pd
import matplotlib.pyplot as plt

# Inicializa
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

def extrair_serie_temporal(lat, lon):
    print(f"📊 Extraindo histórico detalhado para Lat: {lat}, Lon: {lon}...")
    
    ponto = ee.Geometry.Point([lon, lat])

    # Filtrando Sentinel-5P (NO2)
    colecao = (ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
               .filterBounds(ponto)
               .filterDate('2023-01-01', '2025-01-01') 
               .select('NO2_column_number_density'))

    def extrair_valor(imagem):
        data = imagem.date().format("YYYY-MM-dd")
        
        # Tenta reduzir a região. Se for tudo mascarado (nuvem), retorna null
        valor_dict = imagem.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ponto,
            scale=1000
        )
        
        # Pegamos o número. Se não tiver, o GEE retorna null no servidor
        valor = valor_dict.get('NO2_column_number_density')
        
        # Retorna a feature com a propriedade
        return ee.Feature(None, {'data': data, 'poluicao': valor})

    # Trazendo os dados para o computador local
    dados_brutos = colecao.map(extrair_valor).getInfo()

    lista_valores = []
    
    # Processamento robusto
    for item in dados_brutos['features']:
        props = item['properties']
        
        # O método .get evita o KeyError se a chave não existir
        val = props.get('poluicao')
        
        if val is not None:
            lista_valores.append([props['data'], val])

    if len(lista_valores) == 0:
        print("⚠️ Nenhum dado válido encontrado (possivelmente muitas nuvens).")
        return pd.DataFrame()

    df = pd.DataFrame(lista_valores, columns=['Data', 'NO2'])
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data')
    
    return df

# --- Execução ---

df_resultado = extrair_serie_temporal(-23.5505, -46.6333)

if not df_resultado.empty:
    print(f"✅ Dados extraídos com sucesso: {len(df_resultado)} registros.")
    print(df_resultado.head()) 

    # Gerar o Gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(df_resultado['Data'], df_resultado['NO2'], color='purple', linewidth=1, label='NO2 (Sentinel-5P)')
    
    # Adicionando uma média móvel para suavizar o visual (fica mais profissional)
    df_resultado['Media_Movel'] = df_resultado['NO2'].rolling(window=7).mean()
    plt.plot(df_resultado['Data'], df_resultado['Media_Movel'], color='orange', linewidth=2, label='Média Móvel (7 dias)')

    plt.title('Histórico de Poluição (NO2) - São Paulo (2023-2025)')
    plt.xlabel('Data')
    plt.ylabel('Concentração (mol/m²)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
else:
    print("Erro: DataFrame vazio.")