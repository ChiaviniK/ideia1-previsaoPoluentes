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

# --- 2. FUNÇÕES DE DADOS (POLUENTES) ---
@st.cache_data(ttl=3600)
def get_data(lat, lon, gas_type):
    ponto = ee.Geometry.Point([lon, lat])
    
    if 'NO2' in gas_type:
        col_id, band, scale = 'COPERNICUS/S5P/NRTI/L3_NO2', 'NO2_column_number_density', 3000
    elif 'CH4' in gas_type:
        col_id, band, scale = 'COPERNICUS/S5P/OFFL/L3_CH4', 'CH4_column_volume_mixing_ratio_dry_air', 5000
    elif 'CO' in gas_type:
        col_id, band, scale = 'COPERNICUS/S5P/NRTI/L3_CO', 'CO_column_number_density', 3000
    elif 'SO2' in gas_type:
        col_id, band, scale = 'COPERNICUS/S5P/NRTI/L3_SO2', 'SO2_column_number_density', 3000
    else:
        return pd.DataFrame(), None

    collection = (ee.ImageCollection(col_id)
                  .filterBounds(ponto)
                  .filterDate('2022-01-01', '2025-01-01')
                  .select(band))

    def extract(img):
        date = img.date().format("YYYY-MM-dd")
        val = img.reduceRegion(ee.Reducer.mean(), ponto, scale).get(band)
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
            
    return df, band

# --- 3. FUNÇÃO: SAÚDE DA VEGETAÇÃO (NDVI) ---
@st.cache_data(ttl=3600)
def get_ndvi(lat, lon):
    ponto = ee.Geometry.Point([lon, lat])
    collection = (ee.ImageCollection('MODIS/006/MOD13Q1')
                  .filterBounds(ponto)
                  .filterDate('2022-01-01', '2025-01-01')
                  .select('NDVI'))

    def extract(img):
        date = img.date().format("YYYY-MM-dd")
        val = img.reduceRegion(ee.Reducer.mean(), ponto, 1000).get('NDVI')
        return img.set({'ds': date, 'ndvi': val})

    mapped = collection.map(extract).filter(ee.Filter.notNull(['ndvi'])).limit(500, 'system:time_start')
    data = mapped.reduceColumns(ee.Reducer.toList(2), ['ds', 'ndvi']).get('list').getInfo()
    
    df = pd.DataFrame(data, columns=['ds', 'ndvi'])
    if not df.empty:
        df['ds'] = pd.to_datetime(df['ds'])
        df['ndvi'] = df['ndvi'] / 10000 
        df = df.sort_values('ds')
    return df

def run_forecast(df):
    m = Prophet(daily_seasonality=False, weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=365*2) 
    forecast = m.predict(future)
    return forecast

# --- 4. FUNÇÃO: MAPA DE CALOR ---
def get_heatmap_layer(gas_type):
    if 'NO2' in gas_type:
        col_id, band = 'COPERNICUS/S5P/NRTI/L3_NO2', 'NO2_column_number_density'
        vis = {'min': 0, 'max': 0.0002, 'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']}
    elif 'CH4' in gas_type:
        col_id, band = 'COPERNICUS/S5P/OFFL/L3_CH4', 'CH4_column_volume_mixing_ratio_dry_air'
        vis = {'min': 1750, 'max': 1900, 'palette': ['black', 'blue', 'cyan', 'green', 'yellow', 'red']}
    elif 'CO' in gas_type:
        col_id, band = 'COPERNICUS/S5P/NRTI/L3_CO', 'CO_column_number_density'
        vis = {'min': 0, 'max': 0.05, 'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']}
    elif 'SO2' in gas_type:
        col_id, band = 'COPERNICUS/S5P/NRTI/L3_SO2', 'SO2_column_number_density'
        vis = {'min': 0, 'max': 0.0005, 'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']}

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    collection = ee.ImageCollection(col_id).filterDate(start_date, end_date).select(band).mean()
    map_id_dict = ee.Image(collection).getMapId(vis)
    return map_id_dict['tile_fetcher'].url_format

# --- 5. GESTÃO DE ESTADO ---
if 'selected_lat' not in st.session_state:
    st.session_state.selected_lat = None
if 'selected_lon' not in st.session_state:
    st.session_state.selected_lon = None
if 'last_map_click' not in st.session_state:
    st.session_state.last_map_click = None

# --- 6. INTERFACE DO USUÁRIO ---
with st.sidebar:
    st.header("⚙️ Painel de Controle")
    
    tipo_gas = st.radio(
        "Poluente Alvo:",
        ('NO2 (Urbano)', 'CH4 (Metano)', 'CO (Queimadas)', 'SO2 (Indústria)')
    )
    
    # --- VOLTA DAS DESCRIÇÕES EDUCATIVAS ---
    if 'NO2' in tipo_gas:
        st.info("🚗 **NO2 (Dióxido de Nitrogênio):**\n\nPrincipal indicador de trânsito intenso e atividade industrial. Níveis altos causam problemas respiratórios.")
    elif 'CH4' in tipo_gas:
        st.info("🐄 **CH4 (Metano):**\n\nIndicador chave para o Agronegócio (pecuária/arrozais) e Aterros Sanitários. Potencial de efeito estufa 80x maior que o CO2.")
    elif 'CO' in tipo_gas:
        st.info("🔥 **CO (Monóxido de Carbono):**\n\nResultante de combustão incompleta. Excelente 'proxy' para detectar queimadas florestais e fornos a lenha/carvão.")
    elif 'SO2' in tipo_gas:
        st.info("🏭 **SO2 (Dióxido de Enxofre):**\n\nLigado à queima de combustíveis fósseis pesados (diesel marítimo, carvão) e atividade vulcânica.")

    st.divider()
    st.header("🗺️ Visualização")
    modo_escuro = st.toggle("🌙 Modo Escuro", value=True)
    usar_heatmap = st.toggle("🔥 Camada de Calor", value=True)
    
    st.divider()
    st.header("📍 Busca Precisa")
    input_lat = st.number_input("Lat", value=-23.5505, format="%.4f")
    input_lon = st.number_input("Lon", value=-46.6333, format="%.4f")
    
    def atualizar_manual():
        st.session_state.selected_lat = input_lat
        st.session_state.selected_lon = input_lon
        st.session_state.last_map_click = None 

    st.button("🔎 Ir para Coordenada", on_click=atualizar_manual)

st.title(f"🌍 CarbonCast AI: {tipo_gas}")

col_map, col_data = st.columns([1.3, 2])

with col_map:
    center_lat = st.session_state.selected_lat if st.session_state.selected_lat else -15.7975
    center_lon = st.session_state.selected_lon if st.session_state.selected_lon else -47.8919
    zoom = 10 if st.session_state.selected_lat else 4
    
    tile_style = 'CartoDB dark_matter' if modo_escuro else 'CartoDB positron'
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=tile_style)
    
    if usar_heatmap:
        try:
            heatmap_url = get_heatmap_layer(tipo_gas)
            folium.TileLayer(tiles=heatmap_url, attr='ESA/Copernicus', overlay=True, name='Poluição', opacity=0.6).add_to(m)
        except:
            st.warning("Heatmap indisponível.")

    if st.session_state.selected_lat:
        folium.Marker([st.session_state.selected_lat, st.session_state.selected_lon], icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

    m.add_child(folium.LatLngPopup())
    map_output = st_folium(m, height=700, width=None)

if map_output['last_clicked']:
    novo_lat, novo_lon = map_output['last_clicked']['lat'], map_output['last_clicked']['lng']
    if (novo_lat != st.session_state.last_map_click):
        st.session_state.selected_lat = novo_lat
        st.session_state.selected_lon = novo_lon
        st.session_state.last_map_click = novo_lat 
        st.rerun() 

# --- ÁREA DE DADOS E GRÁFICOS ---
with col_data:
    lat_final, lon_final = st.session_state.selected_lat, st.session_state.selected_lon

    if lat_final and lon_final:
        # Configurações de Cor
        if 'NO2' in tipo_gas: cor = '#ff5733'
        elif 'CH4' in tipo_gas: cor = '#28a745'
        elif 'CO' in tipo_gas: cor = '#6f42c1'
        else: cor = '#e0a800'
        
        bg_theme = "plotly_dark" if modo_escuro else "plotly_white"
        font_color = "white" if modo_escuro else "black"

        with st.spinner(f'📡 Analisando Big Data ({tipo_gas})...'):
            try:
                df, band_name = get_data(lat_final, lon_final, tipo_gas)
                
                if df.empty or len(df) < 5:
                    st.warning("Sem dados suficientes. O satélite pode não cobrir esta área com frequência.")
                else:
                    forecast = run_forecast(df)
                    
                    # Cálculo de Métricas
                    valor_hoje = forecast.iloc[-730]['yhat'] # Aproximado (último ano)
                    valor_futuro = forecast.iloc[-1]['yhat'] # Daqui 2 anos
                    delta = ((valor_futuro - valor_hoje) / valor_hoje) * 100
                    
                    # --- DASHBOARD DE MÉTRICAS (VOLTOU!) ---
                    st.subheader("📊 Diagnóstico Executivo")
                    
                    # Layout: Gauge na Esquerda, Números na Direita
                    c_gauge, c_metrics = st.columns([1, 1.5])
                    
                    with c_gauge:
                        max_gauge = df['y'].max() * 1.2
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = valor_hoje,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Risco Instantâneo"},
                            gauge = {
                                'axis': {'range': [0, max_gauge], 'tickwidth': 1},
                                'bar': {'color': font_color},
                                'steps': [
                                    {'range': [0, max_gauge*0.33], 'color': "#28a745"},
                                    {'range': [max_gauge*0.33, max_gauge*0.66], 'color': "#ffc107"},
                                    {'range': [max_gauge*0.66, max_gauge], 'color': "#dc3545"}],
                            }
                        ))
                        fig_gauge.update_layout(height=200, margin=dict(t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': font_color})
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with c_metrics:
                        st.write("Resumo Estatístico:")
                        col_a, col_b = st.columns(2)
                        col_a.metric("Nível Atual Estimado", f"{valor_hoje:.6f}")
                        col_b.metric("Previsão (2 Anos)", f"{valor_futuro:.6f}", delta=f"{delta:.2f}%")
                        
                        if delta < -5:
                            st.success("📉 Tendência de Redução: Positivo para Crédito de Carbono.")
                        elif delta > 5:
                            st.error("📈 Tendência de Alta: Alerta de Emissões.")
                        else:
                            st.info("➡️ Tendência Estável: Monitoramento contínuo recomendado.")

                    st.divider()

                    # --- GRÁFICO 2: PREVISÃO (PROPHET) ---
                    st.subheader("🔮 Tendência e Sazonalidade")
                    st.caption("Este gráfico utiliza IA para separar o comportamento padrão (sazonalidade) da tendência real de poluição.")
                    
                    fig_main = go.Figure()
                    fig_main.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Leitura Satélite (Real)', marker=dict(color='#888', size=2)))
                    fig_main.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Tendência Projetada (IA)', line=dict(color=cor, width=2)))
                    fig_main.add_trace(go.Scatter(
                        x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
                        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
                        fill='toself', fillcolor=cor, opacity=0.1, line=dict(color='rgba(0,0,0,0)'), name='Intervalo de Incerteza'
                    ))
                    fig_main.update_layout(xaxis_title="Linha do Tempo", yaxis_title="Concentração", template=bg_theme, height=350, legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig_main, use_container_width=True)

                    # --- GRÁFICO 3: CORRELAÇÃO VEGETAÇÃO (NDVI) ---
                    st.divider()
                    st.subheader("🌿 Análise Cruzada: Saúde da Floresta")
                    st.caption("Cruzamento de dados: Comparamos se o aumento da poluição coincide com a perda de cobertura vegetal (desmatamento).")
                    
                    with st.spinner("Buscando dados de vegetação (MODIS)..."):
                        df_ndvi = get_ndvi(lat_final, lon_final)
                        if not df_ndvi.empty:
                            fig_dual = go.Figure()
                            # Eixo 1: Poluição
                            fig_dual.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name=f"Poluição ({tipo_gas.split('(')[0]})", line=dict(color=cor)))
                            # Eixo 2: Vegetação
                            fig_dual.add_trace(go.Scatter(x=df_ndvi['ds'], y=df_ndvi['ndvi'], name="Saúde Vegetação (NDVI)", line=dict(color='#00cc96', width=2, dash='dot'), yaxis='y2'))
                            
                            fig_dual.update_layout(
                                title="Poluição vs. Natureza",
                                template=bg_theme,
                                yaxis=dict(title="Nível de Poluição"),
                                yaxis2=dict(title="Índice NDVI (0=Solo, 1=Floresta)", overlaying='y', side='right', range=[0, 1]),
                                legend=dict(orientation="h", y=1.1),
                                height=350
                            )
                            st.plotly_chart(fig_dual, use_container_width=True)
                        else:
                            st.warning("Dados de vegetação não disponíveis para esta área.")

                    # --- DOWNLOAD ---
                    st.divider()
                    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Baixar Dossiê Técnico (CSV)", csv, f'carbon_report_{lat_final}.csv', 'text/csv')

            except Exception as e:
                st.error(f"Erro de Processamento: {e}")
    else:
        st.info("👈 Use o mapa ou digite as coordenadas para iniciar a auditoria.")