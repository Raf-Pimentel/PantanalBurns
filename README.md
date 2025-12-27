Este repositório contém o pipeline de processamento e análise de dados para a pesquisa "A Reproducible Landsat Workflow for Mapping Burn Severity and Recovery in the Pantanal Wetlands". O projeto utiliza séries temporais de imagens Landsat 8/9 (2021-2025) para monitorizar a resiliência ecológica do Pantanal após grandes incêndios.

📋 Sobre a Pesquisa
O estudo foca-se na análise da recuperação da biomassa vegetal pós-fogo através do índice NBR (Normalized Burn Ratio) e da sua correlação com a dinâmica hídrica e de solo exposto medida pelo NDSI (Normalized Difference Snow Index).

A "pitada" de inovação deste fluxo de trabalho é a aplicação de Machine Learning (ARIMA) para realizar o forecasting da tendência de recuperação, permitindo prever se uma área está em trajetória de cura ou de degradação permanente.

🛠️ Funcionalidades Técnicas
Extração Regional Robusta: Cálculo da média espacial de todos os pixels da área de estudo, eliminando ruídos de sensores individuais.

Resiliência de Dados: Tratamento automático de ficheiros Geotiff corrompidos (erros de StripOffsets) e preenchimento de lacunas (Gap Filling) via interpolação temporal.

Machine Learning Preditivo: Implementação de modelo ARIMA para análise de tendências futuras baseadas em 69 pontos temporais.

Visualização Científica: Geração de gráficos de alta resolução (300 DPI) prontos para publicação académica.

🚀 Como Executar
1. Requisitos
É recomendada a utilização de um ambiente virtual para evitar conflitos com o NumPy 2.0:

Bash

python3 -m venv venv
source venv/bin/activate
pip install "numpy<2" pandas rasterio matplotlib scikit-learn statsmodels
2. Estrutura de Dados
O script espera a seguinte estrutura de diretórios:

Plaintext

.
├── PipelinePantanal.py     # Script principal de análise
├── _NBR_OUT/               # Pasta com ficheiros .tif de NBR
│   └── manifest_nbr.csv    # Manifesto com caminhos corrigidos
└── _NDVI_OUT/              # Pasta com ficheiros .tif de NDSI
    └── manifest_ndvi.csv   # Manifesto com caminhos corrigidos
3. Execução
Bash

python3 PipelinePantanal.py
📊 Interpretação dos Resultados
NBR (Linha Verde): Representa a saúde da vegetação. Uma inclinação positiva após 2021 indica recuperação bem-sucedida.

NDSI (Linha Azul): Monitoriza a presença de água/humidade. Picos de NDSI no Pantanal geralmente correlacionam-se com períodos de cheia que podem desacelerar a detecção de biomassa.

Tendência ML (Linha Vermelha): A previsão gerada pelo modelo ARIMA para os meses subsequentes.

👥 Autores
Douglas Bazo de Castro - Geólogo / Investigador Principal (UNICAMP)

Vinícius dos Santos Pereira - Co-autor

Rafael Pimentel - Desenvolvimento de Software e Ciência de Dados