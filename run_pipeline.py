#!/usr/bin/env python3
"""
run_pipeline.py
Script principal para la ejecucion del pipeline integral en la rama `etl_JP`:
1. Extraccion y generacion del dataset tabular de caracteristicas (ETL) a partir de los datos crudos.
2. Carga y preprocesamiento (Z-Score) sin librerias de ML.
3. Entrenamiento, evaluacion y reporte del clasificador k-NN desde cero.
4. Demostracion de predicciones en consola con diagnostico y explicacion de vecinos.

Uso:
    python run_pipeline.py
    python run_pipeline.py --k 7 --metric euclidean --weights distance --demo-samples 8
"""

import sys
import argparse
from pathlib import Path

from src.etl_features import generate_tabular_dataset
from src.knn_from_scratch import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline integral de ETL y Clasificador k-NN Desde Cero (Rama etl_JP)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directorio donde residen los archivos .npy crudos del dataset REHAB."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/rehab_tabular_features.csv",
        help="Ruta de destino/origen del archivo CSV con caracteristicas tabulares."
    )
    parser.add_argument(
        "--force-etl",
        action="store_true",
        help="Fuerza la re-ejecucion del proceso ETL aunque el CSV ya exista."
    )
    parser.add_argument(
        "-k", "--neighbors",
        type=int,
        default=5,
        help="Numero de vecinos k para el clasificador k-NN (default: 5)."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan"],
        help="Metrica de distancia geometrica (default: euclidean)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="distance",
        choices=["distance", "uniform"],
        help="Esquema de ponderacion de votos de los vecinos (default: distance)."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporcion del conjunto de prueba (default: 0.2)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (default: 42)."
    )
    parser.add_argument(
        "--demo-samples",
        type=int,
        default=6,
        help="Numero de muestras de prueba a inspeccionar en detalle por consola."
    )

    args = parser.parse_args()

    csv_file = Path(args.csv_path)

    # 1. Ejecutar ETL si no existe el CSV o si se solicita explicitamente
    if not csv_file.exists() or args.force_etl:
        print("[PIPELINE] Ejecutando etapa de extraccion de caracteristicas (ETL)...")
        generate_tabular_dataset(
            data_dir=args.data_dir,
            output_csv_path=args.csv_path,
            verbose=True
        )
    else:
        print(f"[PIPELINE] Usando dataset tabular existente en: {csv_file.resolve()}")

    # 2. Ejecutar entrenamiento, evaluacion y predicciones
    run_experiment(
        csv_path=args.csv_path,
        k=args.neighbors,
        metric=args.metric,
        weights=args.weights,
        test_size=args.test_size,
        random_state=args.seed,
        num_demo_samples=args.demo_samples
    )


if __name__ == "__main__":
    main()
