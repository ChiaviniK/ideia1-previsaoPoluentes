import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuração de estilo para parecer profissional
plt.style.use('bmh') # Estilo visual limpo (Business/Science)

def plot_analise():
    print("📊 Carregando dados...")
    try:
        df = pd.read_csv("historico_poluicao.csv")
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'historico_poluicao.csv' não encontrado. Rode o script anterior primeiro.")
        return

    # Converter coluna de data (string) para objeto datetime real
    df['data_hora'] = pd.to_datetime(df['data_hora'])

    # --- TÉCNICA DE RESAMPLING ---
    # Dados horários são muito "tremidos". Vamos tirar a média DIÁRIA.
    # Isso suaviza o gráfico e mostra a tendência real.
    df_diario = df.set_index('data_hora').resample('D').mean()

    # Criar uma figura com 2 gráficos (subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # GRÁFICO 1: Poluição por Partículas (PM2.5) - Saúde
    ax1.plot(df_diario.index, df_diario['pm2_5'], color='#d62728', linewidth=2)
    ax1.set_title('Concentração Média Diária de PM2.5 (Risco à Saúde)', fontsize=14)
    ax1.set_ylabel('µg/m³')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar uma linha de "Perigo" (Exemplo da OMS: > 25 é ruim)
    ax1.axhline(y=25, color='black', linestyle='--', label='Limite OMS (24h)')
    ax1.legend()

    # GRÁFICO 2: Dióxido de Nitrogênio (NO2) - Trânsito/Indústria
    ax2.plot(df_diario.index, df_diario['no2'], color='#1f77b4', linewidth=2)
    ax2.set_title('Concentração Média Diária de NO2 (Indicador de Tráfego)', fontsize=14)
    ax2.set_ylabel('µg/m³')
    ax2.set_xlabel('Data')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Formatar eixo X para mostrar os meses corretamente
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    
    print("📈 Gerando gráfico...")
    plt.show() # Abre a janela com o gráfico

if __name__ == "__main__":
    # Garantir que matplotlib esteja instalado
    # pip install matplotlib
    plot_analise()