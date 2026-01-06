import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import warnings
import os

# Disable rasterio metadata warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_raster_stats(path):
    """Reads the raster and extracts statistics from valid pixels in the range [-1, 1]."""
    if not path.exists():
        return None
    try:
        with rasterio.open(str(path)) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            
            # Filter: remove NoData and keep only the physical NDVI/NBR range (-1 to 1)
            # If your data are Landsat scaled (0–10000), change 1.1 to 10001
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
    # 1. Define paths based on your directory tree structure
    base_dir = Path(os.getcwd())
    print(f"--- Starting Processing in the Pantanal ---")
    print(f"Base directory: {base_dir}")

    # Exact locations according to your tree command:
    nbr_manifest = base_dir / "_NBR_OUT" / "manifest_nbr.csv"
    ndvi_manifest = base_dir / "_NDVI_OUT" / "manifest_ndvi.csv"

    # Existence check
    if not nbr_manifest.exists():
        print(f"ERROR: Could not find {nbr_manifest}")
        return
    if not ndvi_manifest.exists():
        print(f"ERROR: Could not find {ndvi_manifest}")
        return

    # 2. Load data
    print("Loading manifests...")
    df_nbr = pd.read_csv(nbr_manifest)
    df_ndvi = pd.read_csv(ndvi_manifest)

    # Merge by scene_id to ensure we compare the same scene over time
    df_merged = pd.merge(
        df_nbr[['scene_id', 'date_acquired', 'nbr_path']], 
        df_ndvi[['scene_id', 'ndvi_path']], 
        on='scene_id'
    )

    df_merged['date_acquired'] = pd.to_datetime(df_merged['date_acquired'])
    df_merged = df_merged.sort_values('date_acquired')
    
    start_date = df_merged['date_acquired'].min()
    results = []

    # 3. Process images
    print(f"Analyzing {len(df_merged)} temporal scenes...")
    for idx, row in df_merged.iterrows():
        # Ignore absolute paths from the CSV and rebuild local paths
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
                print(f"Progress: {idx + 1}/{len(df_merged)} scenes processed.")

    # 4. Save results
    if results:
        output_csv = base_dir / "resultados_temporais.csv"
        df_final = pd.DataFrame(results)
        df_final.to_csv(output_csv, index=False)
        print(f"\n--- SUCCESS ---")
        print(f"Generated file: {output_csv}")
        print(f"Total processed dates: {len(df_final)}")
    else:
        print("\nERROR: No images could be processed. Check whether the .tif files are valid.")

if __name__ == "__main__":
    main()
