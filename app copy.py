import streamlit as st
import ee
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import folium
from streamlit_folium import st_folium
import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="CarbonCast AI Pro", layout="wide", page_icon="🌍")

# --- 1. CONEXÃO COM O GOOGLE EARTH ENGINE ---
@st.cache_resource
def initialize_ee():
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()

initialize_ee()

# --- 2. FUNÇÃO DE DADOS (GRÁFICO E CSV) ---
@st.cache_data(ttl=3600)
def get_data(lat, lon, gas_type):
    ponto = ee.Geometry.Point([lon, lat])
    
    if gas_type == 'NO2':
        collection_id = 'COPERNICUS/S5P/NRTI/L3_NO2'
        band_id = 'NO2_column_number_density'
        escala = 3000
    else: # CH4
        collection_id = 'COPERNICUS/S5P/OFFL/L3_CH4'
        band_id = 'CH4_column_volume_mixing_ratio_dry_air'
        escala = 5000

    collection = (ee.ImageCollection(collection_id)
                  .filterBounds(ponto)
                  .filterDate('2022-01-01', '2025-01-01')
                  .select(band_id))

    def extract(img):
        date = img.date().format("YYYY-MM-dd")
        val = img.reduceRegion(ee.Reducer.mean(), ponto, escala).get(band_id)
        return img.set({'ds': date, 'y': val})

    mapped_col = collection.map(extract)
    clean_col = mapped_col.filter(ee.Filter.notNull(['y'])).limit(1000, 'system:time_start')

    data = clean_col.reduceColumns(ee.Reducer.toList(2), ['ds', 'y']).get('list').getInfo()
    df = pd.DataFrame(data, columns=['ds', 'y'])
    
    if not df.empty:
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        if len(df) > 10:
            q_low = df["y"].quantile(0.01)
            q_hi  = df["y"].quantile(0.99)
            df = df[(df["y"] < q_hi) & (df["y"] > q_low)]
            
    return df, band_id

def run_forecast(df):
    m = Prophet(daily_seasonality=False, weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=365*2) 
    forecast = m.predict(future)
    return forecast

# --- 3. NOVA FUNÇÃO: GERADOR DE MAPA DE CALOR ---
def get_heatmap_layer(gas_type):
    # Define parâmetros visuais baseados no gás
    if gas_type == 'NO2':
        collection_id = 'COPERNICUS/S5P/NRTI/L3_NO2'
        band_id = 'NO2_column_number_density'
        # Cores: Transparente -> Azul -> Roxo -> Vermelho (Perigo)
        vis_params = {
            'min': 0,
            'max': 0.0002,
            'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
        }
    else: # CH4
        collection_id = 'COPERNICUS/S5P/OFFL/L3_CH4'
        band_id = 'CH4_column_volume_mixing_ratio_dry_air'
        # O metano varia muito pouco, ajustamos o min/max para dar contraste
        vis_params = {
            'min': 1750, # Base aproximada em ppb
            'max': 1900,
            'palette': ['black', 'blue', 'cyan', 'green', 'yellow', 'red']
        }

    # Pega apenas os últimos 30 dias para mostrar a "situação atual"
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)

    collection = (ee.ImageCollection(collection_id)
                  .filterDate(start_date, end_date)
                  .select(band_id)
                  .mean()) # Faz a média dos últimos 30 dias em cada pixel

    # Gera o link do "azulejo" (Tile URL) para o Folium usar
    map_id_dict = ee.Image(collection).getMapId(vis_params)
    return map_id_dict['tile_fetcher'].url_format

# --- 4. INTERFACE DO USUÁRIO ---

with st.sidebar:
    st.header("⚙️ Configurações")
    tipo_gas = st.radio("Selecione o Poluente:", ('NO2', 'CH4'))
    
    st.info("💡 **Dica Visual:**\n\nO mapa mostrará as concentrações médias dos últimos 30 dias.\n\n🔴 **Vermelho:** Alta concentração.\n🔵 **Azul/Preto:** Baixa concentração.")

st.title(f"🌍 CarbonCast AI: Heatmap de {tipo_gas}")
st.markdown("O mapa abaixo mostra onde estão as maiores emissões **agora**. Clique em qualquer ponto para ver a previsão futura.")

col_map, col_data = st.columns([1.3, 2]) # Aumentei um pouco a largura do mapa

with col_map:
    st.subheader("📍 Mapa de Calor Global")
    
    # 1. Cria o mapa base (Dark Matter fica melhor para ver as cores brilhantes)
    m = folium.Map(location=[-15.7975, -47.8919], zoom_start=4, tiles='CartoDB dark_matter')
    
    # 2. Gera a camada de calor do Google Earth Engine
    try:
        heatmap_url = get_heatmap_layer(tipo_gas)
        
        # 3. Adiciona a camada ao mapa
        folium.TileLayer(
            tiles=heatmap_url,
            attr='Google Earth Engine & Copernicus',
            overlay=True,
            name=f'Concentração {tipo_gas}',
            opacity=0.7
        ).add_to(m)
        
    except Exception as e:
        st.error(f"Erro ao carregar mapa: {e}")

    # 4. Adiciona controle de camadas e clique
    folium.LayerControl().add_to(m)
    m.add_child(folium.LatLngPopup())
    
    map_output = st_folium(m, height=600, width=500)

lat, lon = None, None
if map_output['last_clicked']:
    lat = map_output['last_clicked']['lat']
    lon = map_output['last_clicked']['lng']

with col_data:
    if lat and lon:
        st.success(f"🔍 Investigando ponto: {lat:.4f}, {lon:.4f}")
        
        with st.spinner(f'📡 Extraindo série temporal de {tipo_gas} e calculando futuro...'):
            try:
                df, band_name = get_data(lat, lon, tipo_gas)
                
                if df.empty or len(df) < 5:
                    st.warning(f"Sem dados suficientes neste pixel específico. Tente clicar exatamente sobre uma mancha colorida.")
                else:
                    forecast = run_forecast(df)
                    
                    fig = go.Figure()
                    cor_linha = '#ff5733' if tipo_gas == 'NO2' else '#28a745'
                    
                    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Histórico', marker=dict(color='#888', size=3)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão AI', line=dict(color=cor_linha, width=3)))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
                        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
                        fill='toself', fillcolor=cor_linha, opacity=0.15,
                        line=dict(color='rgba(0,0,0,0)'), name='Margem de Erro'
                    ))

                    unidade = "mol/m²" if tipo_gas == 'NO2' else "ppbv"
                    fig.update_layout(title=f"Tendência de {tipo_gas} (2022-2027)", xaxis_title="Ano", yaxis_title=unidade, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                    # Métricas
                    v_hoje = forecast.iloc[-730]['yhat']
                    v_fut = forecast.iloc[-1]['yhat']
                    delta = ((v_fut - v_hoje) / v_hoje) * 100
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Nível Atual", f"{v_hoje:.6f}")
                    c2.metric("Previsão (2 Anos)", f"{v_fut:.6f}", delta=f"{delta:.2f}%")

                    # Download
                    st.divider()
                    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Baixar CSV para Auditoria", csv, f'report_{lat}_{lon}.csv', 'text/csv')

            except Exception as e:
                st.error(f"Erro técnico: {e}")
    else:
        st.info("👈 O mapa à esquerda mostra a poluição REAL dos últimos 30 dias.\n\nClique em uma **área vermelha/amarela** para gerar a previsão.")