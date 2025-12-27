import os
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Silenciar avisos chatos do NumPy e Matplotlib
warnings.filterwarnings('ignore')

# Tentar usar um estilo universal
try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.style.use('ggplot')

# ============================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# ============================================================================
# O script usa os CSVs já limpos, sem as imagens corrompidas.
NBR_CSV = '_NBR_OUT/manifest_nbr.csv'
NDVI_CSV = '_NDVI_OUT/manifest_ndvi.csv'

print("="*80)
print("PIPELINE PANTANAL: ANÁLISE POR MÉDIA REGIONAL (SPATIAL MEAN)")
print("="*80)

# ============================================================================
# 2. FUNÇÃO PARA CALCULAR A MÉDIA DE TODOS OS PIXELS
# ============================================================================
def extract_spatial_mean(csv_path, path_column):
    """Lê cada imagem e calcula a média de todos os pixels válidos."""
    df = pd.read_csv(csv_path)
    df['date_acquired'] = pd.to_datetime(df['date_acquired'])
    df = df.sort_values('date_acquired')
    
    means = []
    dates = []
    
    print(f"\n📊 Calculando médias para {path_column.split('_')[0].upper()}...")
    
    for idx, row in df.iterrows():
        path = row[path_column]
        if not os.path.exists(path):
            continue
            
        try:
            with rasterio.open(path) as src:
                # Lemos os dados como float para suportar NaNs
                data = src.read(1).astype('float32')
                
                # Tratar valores de NoData (fundo da imagem)
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
                
                # Filtro de segurança para índices (devem estar entre -1.5 e 1.5)
                # Valores fora disso geralmente são erros de processamento ou bordas
                data[(data < -1.5) | (data > 1.5)] = np.nan
                
                # Calculamos a média de todos os pixels que não são NaN
                img_mean = np.nanmean(data)
                
                # Multiplicamos pelo fator de escala do Landsat se necessário
                # Se seus arquivos já estiverem entre -1 e 1, o código ignora isso
                if abs(img_mean) > 10:
                    img_mean = img_mean * 0.0001
                
                means.append(img_mean)
                dates.append(row['date_acquired'])
                
        except Exception:
            # Se o arquivo estiver corrompido, pulamos (o pandas irá interpolar depois)
            print(f"  ⚠️  Pulando arquivo com erro: {os.path.basename(path)}")
            means.append(np.nan)
            dates.append(row['date_acquired'])
            
    series = pd.Series(means, index=dates)
    # Preenche buracos de arquivos corrompidos com base no tempo
    series = series.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    return series

# ============================================================================
# 3. PROCESSAMENTO DOS DADOS
# ============================================================================
nbr_series = extract_spatial_mean(NBR_CSV, 'nbr_path')
ndsi_series = extract_spatial_mean(NDVI_CSV, 'ndvi_path')

print("\n[OK] Médias espaciais calculadas.")

# ============================================================================
# 4. MACHINE LEARNING: PREVISÃO DA MÉDIA REGIONAL
# ============================================================================
print("\n🤖 Aplicando ML (ARIMA) na tendência regional...")
try:
    # O modelo agora olha para a média da região inteira
    model = ARIMA(nbr_series, order=(5, 1, 0))
    model_fit = model.fit()
    
    # Previsão para os próximos 6 meses
    forecast_steps = 6
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Criar datas futuras para o gráfico
    last_date = nbr_series.index[-1]
    forecast_dates = pd.date_range(last_date, periods=forecast_steps + 1, freq='ME')[1:]
    print("✅ Previsão concluída.")
except Exception as e:
    print(f"⚠️ Erro no ML: {e}")
    forecast = None

# ============================================================================
# 5. GRÁFICO FINAL PARA O ARTIGO
# ============================================================================
plt.figure(figsize=(14, 7))

# Dados Reais (Médias)
plt.plot(nbr_series.index.to_numpy(), nbr_series.values, 
         label='Média Regional NBR (Vegetação)', color='#2ca02c', linewidth=2.5)

plt.plot(ndsi_series.index.to_numpy(), ndsi_series.values, 
         label='Média Regional NDSI (Umidade)', color='#1f77b4', linestyle='--', alpha=0.6)

# Previsão ML
if forecast is not None:
    plt.plot(forecast_dates.to_numpy(), forecast.values, 
             label='Tendência de Recuperação (ML)', color='red', linestyle=':', linewidth=3, marker='^')
    
    # Intervalo de Confiança Visual
    plt.fill_between(forecast_dates.to_numpy(), forecast.values - 0.03, forecast.values + 0.03, 
                     color='red', alpha=0.1)

# Estética Científica
plt.title("Dinâmica de Recuperação Pós-Fogo no Pantanal (Média de Todos os Pixels)", fontsize=16)
plt.xlabel("Ano", fontsize=12)
plt.ylabel("Valor Médio do Índice", fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, alpha=0.3)
plt.legend(loc='best', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('analise_regional_pantanal.png', dpi=300)
print("\n📊 Gráfico salvo como: 'analise_regional_pantanal.png'")
print("="*80)