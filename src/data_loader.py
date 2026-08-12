"""
data_loader.py
Carga y consolidacion del dataset REHAB - Rehab_exercise (16 movimientos de
entrenamiento de rehabilitacion) a partir de los archivos .npy publicados en
Science Data Bank (https://doi.org/10.57760/sciencedb.37018).

Referencia: Lv et al. (2026), "A wearable sensor-based kinematic dataset
collected under standardized rehabilitation tasks from 120 post-stroke
patients", Scientific Data 13:1136.

Estructura esperada de carpeta:

    data/
        000_1.npy   000_2.npy
        001_1.npy   001_2.npy
        ...
        015_1.npy   015_2.npy

Cada archivo <mov>_1.npy tiene forma (d, 880, 6): pitch/yaw/roll de los dos
sensores IMU (brazo+antebrazo o muslo+pantorrilla segun el movimiento).
Cada archivo <mov>_2.npy tiene forma (d, 880, 6): flexion de los 5 dedos +
pitch del guante (sensor S5).

NOTA IMPORTANTE - dato corrupto conocido:
El archivo oficial "014_1.npy" (movimiento 14: flexion/extension de rodilla)
esta corrupto en el repositorio publico de Science Data Bank (el MD5 publicado
coincide con el archivo danado, asi que no es un error de descarga). Solo
tenemos "014_2.npy" (canales del guante) para ese movimiento. Por defecto,
build_dataset() excluye el movimiento 14 del set principal de 12 canales y
lo deja disponible aparte con solo 6 canales via la bandera
`incluir_mov14_parcial=True`.
"""

from pathlib import Path
import numpy as np

N_MOVEMENTS = 16
SIGNAL_LENGTH = 880
MISSING_FILES = {(14, 1)}  # (movimiento, sensorID) conocido como corrupto/ausente

MOVEMENT_NAMES = {
    0: "bobath handshake",
    1: "bobath flexion/extension",
    2: "bobath forward flexion/extension",
    3: "bobath anterior/posterior rotation",
    4: "elbow flexion and wrist compression",
    5: "wrist flexion and extension",
    6: "finger-to-finger training",
    7: "ball gripping",
    8: "shoulder joint internal and external rotation",
    9: "breast expansion",
    10: "flexion-pressure rotation forward and backward",
    11: "elbow joint flexion and touch",
    12: "shoulder touch training",
    13: "ankle extension & knee internal/external rotation",
    14: "knee flexion and extension",
    15: "hip flexion and extension",
}


def _load_pair(base_path: Path, movement_id: int):
    """Carga (si existen) los archivos <mov>_1.npy y <mov>_2.npy de un movimiento."""
    mov_str = f"{movement_id:03d}"
    arr1 = arr2 = None

    f1 = base_path / f"{mov_str}_1.npy"
    if f1.exists() and (movement_id, 1) not in MISSING_FILES:
        arr1 = np.load(f1)

    f2 = base_path / f"{mov_str}_2.npy"
    if f2.exists() and (movement_id, 2) not in MISSING_FILES:
        arr2 = np.load(f2)

    return arr1, arr2


def build_dataset(data_dir: str, incluir_mov14_parcial: bool = False, verbose: bool = True):
    """
    Consolida todos los movimientos disponibles en un solo dataset.

    Parameters
    ----------
    data_dir : str
        Carpeta donde estan los .npy (ej. "data/" o la ruta a
        Rehab_exercise/d02_processed_data despues de montar Google Drive).
    incluir_mov14_parcial : bool
        Si True, agrega tambien el movimiento 14 usando solo los 6 canales
        del guante (rellenando con NaN los 6 canales IMU faltantes) para
        no perder esas muestras. Si False (default), el movimiento 14 se
        omite del dataset principal de 12 canales.
    verbose : bool
        Imprime un resumen de lo que se cargo.

    Returns
    -------
    X : np.ndarray, forma (N, 880, 12)
        Senal consolidada: canales 0-5 = IMU (pitch/yaw/roll x2 sensores),
        canales 6-11 = guante (f1..f5, pitch3).
    y : np.ndarray, forma (N,)
        Etiqueta de movimiento (0-15).
    meta : dict
        Info por movimiento: cuantas muestras y que sensores se encontraron.
    """
    base_path = Path(data_dir)
    X_parts, y_parts = [], []
    meta = {}

    for mov in range(N_MOVEMENTS):
        arr1, arr2 = _load_pair(base_path, mov)
        meta[mov] = {
            "nombre": MOVEMENT_NAMES[mov],
            "sensor_imu_disponible": arr1 is not None,
            "sensor_guante_disponible": arr2 is not None,
        }

        if arr1 is not None and arr2 is not None:
            if arr1.shape[0] != arr2.shape[0]:
                raise ValueError(
                    f"Movimiento {mov}: numero de muestras distinto entre "
                    f"sensor 1 ({arr1.shape[0]}) y sensor 2 ({arr2.shape[0]})"
                )
            X_mov = np.concatenate([arr1, arr2], axis=-1)  # (d, 880, 12)
            meta[mov]["muestras"] = X_mov.shape[0]
        elif arr2 is not None and incluir_mov14_parcial:
            d = arr2.shape[0]
            relleno = np.full((d, SIGNAL_LENGTH, 6), np.nan)
            X_mov = np.concatenate([relleno, arr2], axis=-1)  # (d, 880, 12) con NaN
            meta[mov]["muestras"] = d
            meta[mov]["advertencia"] = "canales IMU rellenados con NaN (014_1.npy corrupto en la fuente)"
        else:
            meta[mov]["muestras"] = 0
            if arr1 is None and (mov, 1) in MISSING_FILES:
                meta[mov]["advertencia"] = "excluido: 014_1.npy corrupto en Science Data Bank"
            continue

        X_parts.append(X_mov)
        y_parts.append(np.full(X_mov.shape[0], mov, dtype=int))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if verbose:
        print(f"Dataset consolidado: X={X.shape}, y={y.shape}")
        print(f"Clases presentes: {sorted(set(y.tolist()))}")
        for mov, info in meta.items():
            flag = "" if info["muestras"] > 0 else "  <-- SIN DATOS"
            print(f"  mov {mov:02d} ({info['nombre']}): {info['muestras']} muestras{flag}")

    return X, y, meta


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    build_dataset(data_dir)
