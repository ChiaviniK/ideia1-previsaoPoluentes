import ee
import datetime

# 1. Autenticação e Inicialização
# Na primeira vez que rodar, isso vai abrir uma janela no navegador pedindo permissão.
# Depois de dar permissão, ele gera um token.
try:
    ee.Initialize()
except Exception as e:
    print("A inicializar autenticação...")
    ee.Authenticate()
    ee.Initialize()

def obter_poluicao(lat, lon):
    print(f"🔄 Consultando satélite para Lat: {lat}, Lon: {lon}...")

    # 2. Definir o local (Ponto geográfico)
    ponto = ee.Geometry.Point([lon, lat]) # Atenção: GEE usa [Longitude, Latitude]

    # 3. Acessar a Coleção do Sentinel-5P (Nível 2 - Dióxido de Nitrogênio)
    # COPERNICUS/S5P/NRTI/L3_NO2 é o ID da coleção de dados em tempo real/quase real
    colecao = (ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
               .filterBounds(ponto)
               .filterDate('2024-01-01', '2024-06-01') # Intervalo de tempo
               .select('NO2_column_number_density')) # Selecionar apenas a banda de NO2

    # 4. Verificar se achou imagens
    qtd_imagens = colecao.size().getInfo()
    print(f"📡 Imagens de satélite encontradas no período: {qtd_imagens}")

    if qtd_imagens == 0:
        return "Nenhum dado encontrado para este período/local."

    # 5. Redução (Cálculo da Média)
    # Pega todas as imagens do período e cria uma única imagem com a média dos valores
    imagem_media = colecao.mean()

    # 6. Extrair o valor numérico exato no ponto
    # Scale: 1000 metros (resolução aproximada do satélite para extração rápida)
    dados = imagem_media.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ponto,
        scale=1000 
    ).getInfo()

    # O valor vem em mol/m^2. É um número muito pequeno (ex: 0.00012).
    # Para facilitar a leitura humana, multiplicamos por 1 milhão ou convertemos.
    valor_no2 = dados.get('NO2_column_number_density')
    
    if valor_no2:
        return f"Concentração Média de NO2: {valor_no2:.6f} mol/m²"
    else:
        return "Dado indisponível (coberto por nuvens ou fora da varredura)."

# --- TESTE ---
# Exemplo: São Paulo, Brasil (Alta poluição esperada)
lat_sp = -23.5505
lon_sp = -46.6333

resultado = obter_poluicao(lat_sp, lon_sp)
print("\n" + "="*30)
print(resultado)
print("="*30)