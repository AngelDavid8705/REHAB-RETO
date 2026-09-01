"""
etl_features.py
Extraccion, Transformacion y Carga (ETL) para convertir las senales temporales
multidimensionales de sensores (N, 880, 12) del dataset REHAB en un conjunto de
datos tabular (N, 72) apropiado para modelos clasicos de Machine Learning.

Cada muestra temporal de 880 puntos de tiempo por 12 canales se resume mediante
6 estadisticas fundamentales por canal:
1. Media (mean): Posicion o angulo medio del segmento corporal.
2. Desviacion estandar (std): Grado de dinamismo o variabilidad del movimiento.
3. Minimo (min): Extremo inferior alcanzado por el sensor.
4. Maximo (max): Extremo superior alcanzado por el sensor.
5. Rango (range = max - min): Amplitud total del movimiento articular (ROM).
6. RMS (root mean square): Energia / potencia cuadratica de la senal.

Total de caracteristicas = 12 canales x 6 estadisticas = 72 variables predictoras.
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np

# Permite importar data_loader si se ejecuta desde src o desde raiz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import build_dataset, MOVEMENT_NAMES

CHANNEL_NAMES = [
    "imu1_pitch",    # Ch 0: IMU 1 (Brazo / Muslo) Pitch
    "imu1_yaw",      # Ch 1: IMU 1 (Brazo / Muslo) Yaw
    "imu1_roll",     # Ch 2: IMU 1 (Brazo / Muslo) Roll
    "imu2_pitch",    # Ch 3: IMU 2 (Antebrazo / Pantorrilla) Pitch
    "imu2_yaw",      # Ch 4: IMU 2 (Antebrazo / Pantorrilla) Yaw
    "imu2_roll",     # Ch 5: IMU 2 (Antebrazo / Pantorrilla) Roll
    "glove_f1_thumb",   # Ch 6: Flexion Dedo 1 (Pulgar)
    "glove_f2_index",   # Ch 7: Flexion Dedo 2 (Indice)
    "glove_f3_middle",  # Ch 8: Flexion Dedo 3 (Medio)
    "glove_f4_ring",    # Ch 9: Flexion Dedo 4 (Anular)
    "glove_f5_pinky",   # Ch 10: Flexion Dedo 5 (Menique)
    "glove_s5_pitch",   # Ch 11: Guante Sensor S5 Pitch
]

STAT_NAMES = ["mean", "std", "min", "max", "range", "rms"]


def get_feature_names():
    """Genera la lista ordenada de los nombres de las 72 caracteristicas."""
    names = []
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        for stat in STAT_NAMES:
            names.append(f"ch{ch_idx:02d}_{ch_name}_{stat}")
    return names


def extract_features_from_tensor(X_tensor: np.ndarray) -> np.ndarray:
    """
    Extrae las 72 caracteristicas estadisticas a partir del tensor 3D de senales.

    Parameters
    ----------
    X_tensor : np.ndarray de forma (N, 880, 12)
        Senales temporales de sensores.

    Returns
    -------
    X_tab : np.ndarray de forma (N, 72)
        Matriz de caracteristicas tabulares.
    """
    N, T, C = X_tensor.shape
    assert C == 12, f"Se esperaban 12 canales, pero se recibieron {C}"

    # Calculo vectorizado sobre el eje temporal (axis=1)
    mean_vals = np.mean(X_tensor, axis=1)                  # (N, 12)
    std_vals = np.std(X_tensor, axis=1)                    # (N, 12)
    min_vals = np.min(X_tensor, axis=1)                    # (N, 12)
    max_vals = np.max(X_tensor, axis=1)                    # (N, 12)
    range_vals = max_vals - min_vals                       # (N, 12)
    rms_vals = np.sqrt(np.mean(X_tensor ** 2, axis=1))     # (N, 12)

    # Concatenamos de forma estructurada para cada canal: mean, std, min, max, range, rms
    channel_features = []
    for ch in range(C):
        ch_feat = np.column_stack([
            mean_vals[:, ch],
            std_vals[:, ch],
            min_vals[:, ch],
            max_vals[:, ch],
            range_vals[:, ch],
            rms_vals[:, ch],
        ])  # (N, 6)
        channel_features.append(ch_feat)

    X_tab = np.hstack(channel_features)  # (N, 72)
    return X_tab


def generate_tabular_dataset(
    data_dir: str = "data",
    output_csv_path: str = "data/rehab_tabular_features.csv",
    verbose: bool = True
):
    """
    Pipeline principal de ETL:
    1. Carga los archivos .npy usando data_loader.
    2. Extrae las caracteristicas estadisticas.
    3. Exporta el archivo CSV estructurado con encabezados y metadatos.
    """
    if verbose:
        print("=" * 70)
        print("INICIANDO PROCESO ETL: EXTRACCION DE CARACTERISTICAS TABULARES")
        print("=" * 70)
        print(f"Directorio de datos fuente: {data_dir}")

    # 1. Carga de datos
    X_raw, y_raw, meta = build_dataset(data_dir, incluir_mov14_parcial=False, verbose=verbose)
    N = X_raw.shape[0]

    # 2. Extraccion de features
    if verbose:
        print("\nExtrayendo caracteristicas descriptivas por canal...")
    X_features = extract_features_from_tensor(X_raw)
    feature_names = get_feature_names()

    if verbose:
        print(f"Matriz de caracteristicas generada: {X_features.shape} (muestras x variables)")

    # 3. Guardado en CSV
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["sample_id"] + feature_names + ["movement_id", "movement_name"]

    if verbose:
        print(f"Guardando dataset procesado en: {output_path.resolve()}")

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(N):
            mov_id = int(y_raw[i])
            mov_name = MOVEMENT_NAMES.get(mov_id, f"mov_{mov_id}")
            row = [i] + [f"{val:.6f}" for val in X_features[i]] + [mov_id, mov_name]
            writer.writerow(row)

    if verbose:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Archivo generado exitosamente ({file_size_mb:.2f} MB)")
        print(f"Total de registros: {N}")
        print(f"Total de columnas: {len(header)} (1 id + 72 features + 2 target/metadatos)")
        print("=" * 70)

    return X_features, y_raw, feature_names


def load_tabular_csv(csv_path: str = "data/rehab_tabular_features.csv"):
    """
    Carga el dataset tabular desde el archivo CSV utilizando unicamente Python estandar y NumPy.

    Returns
    -------
    X : np.ndarray de forma (N, 72)
    y : np.ndarray de forma (N,)
    feature_names : list[str]
    sample_ids : list[int]
    movement_names : list[str]
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"No se encontro el archivo {csv_path}. Ejecuta generate_tabular_dataset() primero.")

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[1:-2]

        sample_ids = []
        X_rows = []
        y_rows = []
        mov_names = []

        for row in reader:
            sample_ids.append(int(row[0]))
            X_rows.append([float(v) for v in row[1:-2]])
            y_rows.append(int(row[-2]))
            mov_names.append(row[-1])

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.int64)

    return X, y, feature_names, sample_ids, mov_names


if __name__ == "__main__":
    data_dir_arg = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_arg = sys.argv[2] if len(sys.argv) > 2 else "data/rehab_tabular_features.csv"
    generate_tabular_dataset(data_dir=data_dir_arg, output_csv_path=output_arg)
