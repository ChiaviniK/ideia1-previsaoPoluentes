import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configurações
DIAS_HISTORICO = 365
DATA_INICIO = datetime(2023, 1, 1)

print("🛠️ Gerando dados sintéticos de poluição para desenvolvimento...")

# Criar intervalo de datas (hora em hora)
datas = [DATA_INICIO + timedelta(hours=x) for x in range(DIAS_HISTORICO * 24)]
n_registros = len(datas)

# Simular dados (com padrões realistas)
# Vamos usar funções seno/cosseno para simular dia/noite e aleatoriedade
np.random.seed(42) # Para garantir que os dados sejam sempre os mesmos

# Simulação de NO2 (Dióxido de Nitrogênio) - Alto impacto do trânsito
# Picos as 8h e as 18h
horas = np.array([d.hour for d in datas])
padrao_transito = (np.sin((horas - 8) * np.pi / 12)**2 + np.sin((horas - 18) * np.pi / 12)**2) 
no2 = 20 + (50 * padrao_transito) + np.random.normal(0, 10, n_registros)
no2 = np.maximum(no2, 0) # Não permitir valores negativos

# Simulação de PM2.5 (Partículas finas)
# Tende a acumular se não ventar, varia com estação
pm2_5 = 15 + np.random.normal(0, 5, n_registros) + (no2 * 0.3)

# Simulação de AQI (Índice de Qualidade do Ar)
# Baseado no PM2.5 (Simplificação)
conditions = [
    (pm2_5 <= 12),
    (pm2_5 > 12) & (pm2_5 <= 35),
    (pm2_5 > 35) & (pm2_5 <= 55),
    (pm2_5 > 55) & (pm2_5 <= 150),
    (pm2_5 > 150)
]
choices = [1, 2, 3, 4, 5] # 1=Bom, 5=Péssimo
aqi = np.select(conditions, choices, default=5)

# Montar o DataFrame igual ao que viria da API
df_mock = pd.DataFrame({
    'data_hora': datas,
    'aqi': aqi,
    'co': np.random.uniform(200, 500, n_registros),
    'no': np.random.uniform(0, 10, n_registros),
    'no2': no2,
    'o3': np.random.uniform(20, 100, n_registros),
    'so2': np.random.uniform(0, 20, n_registros),
    'pm2_5': pm2_5,
    'pm10': pm2_5 * 1.5, # Geralmente PM10 é maior que PM2.5
    'nh3': np.random.uniform(0, 5, n_registros)
})

# Salvar
arquivo_saida = "historico_poluicao.csv"
df_mock.to_csv(arquivo_saida, index=False)
print(f"✅ Arquivo '{arquivo_saida}' gerado com sucesso!")
print(f"📊 Total de registros: {n_registros}")
print("🚀 Agora você pode prosseguir para a etapa de Previsão/Gráficos.")