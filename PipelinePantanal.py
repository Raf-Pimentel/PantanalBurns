import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import warnings
import os

# Desativar avisos de metadados do rasterio
warnings.filterwarnings("ignore", category=UserWarning)

def get_raster_stats(path):
    """Lê o raster e extrai estatísticas de pixels válidos no intervalo [-1, 1]."""
    if not path.exists():
        return None
    try:
        with rasterio.open(str(path)) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            
            # Filtro: Remove NoData e mantém apenas o range físico de NDVI/NBR (-1 a 1)
            # Se seus dados forem Landsat Scale (0-10000), mude 1.1 para 10001
            mask = (data > -1.1) & (data < 1.1)
            if nodata is not None:
                mask &= (data != nodata)

            valid_data = data[mask]
            if valid_data.size == 0:
                return None
            
            return {
                'mean': np.nanmean(valid_data),
                'std': np.nanstd(valid_data),
                'count': valid_data.size
            }
    except Exception:
        return None

def main():
    # 1. Definir caminhos baseados na estrutura do seu 'tree'
    base_dir = Path(os.getcwd())
    print(f"--- Iniciando Processamento no Pantanal ---")
    print(f"Diretório base: {base_dir}")

    # Localização exata conforme seu comando tree:
    nbr_manifest = base_dir / "_NBR_OUT" / "manifest_nbr.csv"
    ndvi_manifest = base_dir / "_NDVI_OUT" / "manifest_ndvi.csv"

    # Verificação de existência
    if not nbr_manifest.exists():
        print(f"ERRO: Não encontrei {nbr_manifest}")
        return
    if not ndvi_manifest.exists():
        print(f"ERRO: Não encontrei {ndvi_manifest}")
        return

    # 2. Carregar dados
    print("Carregando manifestos...")
    df_nbr = pd.read_csv(nbr_manifest)
    df_ndvi = pd.read_csv(ndvi_manifest)

    # Merge por scene_id para garantir que comparamos a mesma foto no tempo
    df_merged = pd.merge(
        df_nbr[['scene_id', 'date_acquired', 'nbr_path']], 
        df_ndvi[['scene_id', 'ndvi_path']], 
        on='scene_id'
    )

    df_merged['date_acquired'] = pd.to_datetime(df_merged['date_acquired'])
    df_merged = df_merged.sort_values('date_acquired')
    
    start_date = df_merged['date_acquired'].min()
    results = []

    # 3. Processar imagens
    print(f"Analisando {len(df_merged)} cenas temporais...")
    for idx, row in df_merged.iterrows():
        # Ignoramos o caminho absoluto do CSV e montamos o caminho local
        file_nbr = base_dir / "_NBR_OUT" / Path(row['nbr_path']).name
        file_ndvi = base_dir / "_NDVI_OUT" / Path(row['ndvi_path']).name

        stats_nbr = get_raster_stats(file_nbr)
        stats_ndvi = get_raster_stats(file_ndvi)

        if stats_nbr and stats_ndvi:
            results.append({
                'data': row['date_acquired'].strftime('%Y-%m-%d'),
                'dias_desde_inicio': (row['date_acquired'] - start_date).days,
                'media_NBR': stats_nbr['mean'],
                'std_NBR': stats_nbr['std'],
                'media_NDVI': stats_ndvi['mean'],
                'std_NDVI': stats_ndvi['std'],
                'pixels_validos': stats_nbr['count'],
                'scene_id': row['scene_id']
            })
            if (idx + 1) % 5 == 0:
                print(f"Progresso: {idx + 1}/{len(df_merged)} cenas processadas.")

    # 4. Salvar resultados
    if results:
        output_csv = base_dir / "resultados_temporais.csv"
        df_final = pd.DataFrame(results)
        df_final.to_csv(output_csv, index=False)
        print(f"\n--- SUCESSO ---")
        print(f"Arquivo gerado: {output_csv}")
        print(f"Total de datas processadas: {len(df_final)}")
    else:
        print("\nERRO: Nenhuma imagem pôde ser processada. Verifique se os arquivos .tif estão íntegros.")

if __name__ == "__main__":
    main()