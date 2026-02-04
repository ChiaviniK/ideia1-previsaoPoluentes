import streamlit as st
import pandas as pd
from prophet import Prophet
import folium
from streamlit_folium import st_folium

# 1. Configuração da Página
st.set_page_config(page_title="CarbonCast Pro", layout="wide")

st.title("🌍 CarbonCast: Monitoramento e Previsão de Poluentes")

# 2. Sidebar para Input
with st.sidebar:
    st.header("Configuração de Localização")
    lat = st.number_input("Latitude", value=-23.5505) # Ex: São Paulo
    lon = st.number_input("Longitude", value=-46.6333)
    pollutant = st.selectbox("Poluente Alvo", ["CO2", "Metano (CH4)", "NO2"])
    
    st.info("Para Crédito Azul, selecione áreas costeiras.")

# 3. Mapa Interativo (Selecionar ponto)
m = folium.Map(location=[lat, lon], zoom_start=10)
folium.Marker([lat, lon], tooltip="Local Analisado").add_to(m)
st_folium(m, height=300)

# 4. Função Simulada de Coleta de Dados (Aqui entraria a API do Google Earth Engine)
def get_satellite_data(lat, lon, pollutant):
    # Simulação de dados históricos
    dates = pd.date_range(start='2020-01-01', end='2025-01-01', freq='M')
    # Dados aleatórios com tendência de alta (simulando poluição)
    values = [x + (x*0.05) for x in range(len(dates))] 
    df = pd.DataFrame({'ds': dates, 'y': values})
    return df

# 5. Processamento e IA
if st.button("Gerar Análise e Previsão"):
    with st.spinner('Consultando satélites e processando modelos...'):
        
        # A. Coleta
        df_history = get_satellite_data(lat, lon, pollutant)
        
        # B. Previsão com Prophet
        model = Prophet()
        model.fit(df_history)
        future = model.make_future_dataframe(periods=365*2) # Prever 2 anos
        forecast = model.predict(future)
        
        # C. Visualização dos Resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Histórico de Emissões")
            st.line_chart(df_history.set_index('ds'))
            
        with col2:
            st.subheader(f"Previsão para {pollutant} (Próximos 2 anos)")
            st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))

        # D. Análise de Crédito de Carbono (Lógica Simplificada)
        st.divider()
        st.header("Análise de Potencial de Crédito")
        
        # Aqui você usaria uma verificação real de mapa
        if pollutant == "CO2":
            st.success("Detectamos potencial para **Carbono Verde**. A área possui densidade vegetativa...")
        elif pollutant == "Metano (CH4)":
            st.warning("Altas concentrações de Metano. Foco deve ser na redução (Aterros/Agricultura).")