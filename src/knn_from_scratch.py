"""
knn_from_scratch.py
Implementacion de k-Nearest Neighbors (k-NN) y herramientas de evaluacion
completamente DESDE CERO en Python puro y operaciones matriciales basicas con NumPy,
sin utilizar NINGUNA biblioteca de Machine Learning (como scikit-learn, scipy.spatial, etc.).

Modulos implementados manualmente:
1. StandardScalerManual: Normalizacion Z-Score (media=0, std=1) sin data leakage.
2. train_test_split_manual: Division estratificada de datos en Train / Test con semilla.
3. KNNClassifierManual: Clasificador multiclase con distancia Euclidiana/Manhattan,
   soporte para pesos uniformes o por distancia inversa, y calculo de probabilidades.
4. Metricas de evaluacion manuales:
   - Accuracy global
   - Precision, Recall y F1-Score por clase (Macro y Weighted)
   - Matriz de confusion formateada en ASCII
5. Modulo de inferencia y prediccion paso a paso por consola.
"""

import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# Permite importar data_loader si se ejecuta desde src o desde raiz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.etl_features import load_tabular_csv, generate_tabular_dataset
from src.data_loader import MOVEMENT_NAMES


# ==============================================================================
# 1. ESCALADOR ESTANDAR MANUAL (Z-SCORE)
# ==============================================================================

class StandardScalerManual:
    """
    Estandarizador de caracteristicas mediante Z-Score implementado desde cero:
        z = (x - mu) / (sigma + eps)
    
    Ajusta la media y desviacion estandar unicamente en el conjunto de entrenamiento
    para evitar fuga de informacion (data leakage).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "StandardScalerManual":
        """Calcula la media y desviacion estandar de cada columna en X_train."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        # Evitar division por cero en features constantes
        self.std_[self.std_ < self.eps] = 1.0
        self.n_features_ = X_arr.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica la transformacion Z-Score usando los parametros aprendidos."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("El escalador debe ser ajustado con fit() antes de transformar.")
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta y transforma en un solo paso."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Revierte la estandarizacion a la escala original."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("El escalador no ha sido ajustado.")
        Z_arr = np.asarray(Z, dtype=np.float64)
        return Z_arr * (self.std_ + self.eps) + self.mean_


# ==============================================================================
# 2. DIVISION TRAIN / TEST ESTRATIFICADA MANUAL
# ==============================================================================

def train_test_split_manual(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba de manera estratificada
    para conservar la proporcion de clases en ambos subconjuntos.

    Returns
    -------
    X_train, X_test, y_train, y_test, idx_train, idx_test
    """
    assert 0.0 < test_size < 1.0, "test_size debe ser un valor entre 0 y 1."
    N = len(y)
    
    rng = np.random.default_rng(seed=random_state)
    
    if not stratify:
        indices = np.arange(N)
        rng.shuffle(indices)
        n_test = int(N * test_size)
        idx_test = indices[:n_test]
        idx_train = indices[n_test:]
    else:
        unique_classes = np.unique(y)
        idx_train_list = []
        idx_test_list = []

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            rng.shuffle(cls_indices)
            n_cls_test = max(1, int(round(len(cls_indices) * test_size)))
            
            idx_test_list.append(cls_indices[:n_cls_test])
            idx_train_list.append(cls_indices[n_cls_test:])

        idx_train = np.concatenate(idx_train_list)
        idx_test = np.concatenate(idx_test_list)

        # Barajar para mezclar las clases
        rng.shuffle(idx_train)
        rng.shuffle(idx_test)

    return X[idx_train], X[idx_test], y[idx_train], y[idx_test], idx_train, idx_test


# ==============================================================================
# 3. CLASIFICADOR k-NEAREST NEIGHBORS (k-NN) DESDE CERO
# ==============================================================================

class KNNClassifierManual:
    """
    Clasificador k-Nearest Neighbors (k-NN) multiclase desarrollado completamente
    desde cero.

    Parametros
    ----------
    k : int, default=5
        Numero de vecinos mas cercanos a considerar para la votacion.
    metric : str, default='euclidean'
        Metrica de distancia: 'euclidean' (L2) o 'manhattan' (L1).
    weights : str, default='distance'
        Ponderacion de votos:
        - 'uniform': Cada uno de los k vecinos tiene un voto de peso igual (1.0).
        - 'distance': El voto de cada vecino se pondera por la inversa de su distancia (1 / (d + eps)).
    """

    def __init__(self, k: int = 5, metric: str = "euclidean", weights: str = "distance"):
        if k < 1:
            raise ValueError("k debe ser un entero positivo >= 1.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(f"Metrica no soportada: {metric}. Use 'euclidean' o 'manhattan'.")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"Esquema de pesos no soportado: {weights}. Use 'uniform' o 'distance'.")

        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierManual":
        """Almacena el conjunto de entrenamiento de referencia."""
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _compute_distances_vectorized(self, X_query: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de distancias entre las muestras de consulta (M, D)
        y todas las muestras de entrenamiento (N, D).
        
        Retorna matriz de forma (M, N).
        """
        if self.metric == "euclidean":
            # Expansion de binomio para velocidad: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
            # O calculo por diferencias en bloques:
            # X_query: (M, D), X_train: (N, D)
            # Para evitar sobrecarga de memoria, calculamos:
            dists = np.sqrt(
                np.sum(X_query**2, axis=1, keepdims=True)
                + np.sum(self.X_train_**2, axis=1, keepdims=True).T
                - 2.0 * np.dot(X_query, self.X_train_.T)
            )
            # Corregir posibles inconsistencias numericas negativas microscopicas
            dists = np.nan_to_num(dists, nan=0.0, posinf=1e9, neginf=0.0)
            return np.maximum(dists, 0.0)
        
        elif self.metric == "manhattan":
            # Manhattan: sum |x_i - y_i|
            # Calculo por bloque
            M = X_query.shape[0]
            N = self.X_train_.shape[0]
            dists = np.zeros((M, N), dtype=np.float64)
            for i in range(M):
                dists[i, :] = np.sum(np.abs(self.X_train_ - X_query[i, :]), axis=1)
            return dists

        raise ValueError(f"Metrica {self.metric} no implementada.")

    def predict_single(
        self,
        x_sample: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]:
        """
        Realiza la prediccion para un solo vector de caracteristicas y retorna
        la explicacion detallada del vecindario.

        Returns
        -------
        pred_class : int
            Clase predicha.
        neighbor_indices : np.ndarray
            Indices de los k vecinos mas cercanos en X_train.
        neighbor_dists : np.ndarray
            Distancias correspondientes a los k vecinos.
        neighbor_labels : np.ndarray
            Etiquetas de clase de los k vecinos.
        class_probabilities : dict
            Probabilidad normalizada por clase segun los votos ponderados.
        """
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llame a fit() primero.")

        k_val = k if k is not None else self.k
        x_arr = np.asarray(x_sample, dtype=np.float64).reshape(1, -1)
        dists = self._compute_distances_vectorized(x_arr)[0]  # (N,)

        # Obtener los k indices con menor distancia
        k_indices = np.argpartition(dists, k_val)[:k_val]
        # Ordenar exactamente los k primeros
        sorted_order = np.argsort(dists[k_indices])
        neighbor_indices = k_indices[sorted_order]
        neighbor_dists = dists[neighbor_indices]
        neighbor_labels = self.y_train_[neighbor_indices]

        # Votacion
        vote_counts: Dict[int, float] = {cls: 0.0 for cls in self.classes_}
        eps = 1e-8

        for d, label in zip(neighbor_dists, neighbor_labels):
            weight = (1.0 / (d + eps)) if self.weights == "distance" else 1.0
            vote_counts[label] += weight

        total_weight = sum(vote_counts.values()) + eps
        probabilities = {cls: count / total_weight for cls, count in vote_counts.items()}

        pred_class = max(vote_counts.keys(), key=lambda c: vote_counts[c])
        return pred_class, neighbor_indices, neighbor_dists, neighbor_labels, probabilities

    def predict(self, X: np.ndarray, k: Optional[int] = None, batch_size: int = 250) -> np.ndarray:
        """
        Predice las etiquetas para una matriz de caracteristicas X (M, D).
        Usa procesamiento por lotes para optimizar memoria y tiempo de calculo.
        """
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llame a fit() primero.")

        k_val = k if k is not None else self.k
        X_arr = np.asarray(X, dtype=np.float64)
        M = X_arr.shape[0]
        y_pred = np.zeros(M, dtype=np.int64)

        for start_idx in range(0, M, batch_size):
            end_idx = min(start_idx + batch_size, M)
            X_batch = X_arr[start_idx:end_idx]
            
            dists_batch = self._compute_distances_vectorized(X_batch)  # (batch_len, N)
            batch_len = dists_batch.shape[0]

            for i in range(batch_len):
                dists_i = dists_batch[i]
                k_idx = np.argpartition(dists_i, k_val)[:k_val]
                k_dists = dists_i[k_idx]
                k_labels = self.y_train_[k_idx]

                vote_counts: Dict[int, float] = {}
                for d, lbl in zip(k_dists, k_labels):
                    w = (1.0 / (d + 1e-8)) if self.weights == "distance" else 1.0
                    vote_counts[lbl] = vote_counts.get(lbl, 0.0) + w

                pred_lbl = max(vote_counts.keys(), key=lambda c: vote_counts[c])
                y_pred[start_idx + i] = pred_lbl

        return y_pred


# ==============================================================================
# 4. METRICAS DE EVALUACION DESDE CERO
# ==============================================================================

def accuracy_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la proporcion de aciertos totales."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix_manual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Construye la matriz de confusion cuadrada (n_clases, n_clases) donde:
    Fila i = Clase Verdadera
    Columna j = Clase Predicha
    """
    if labels is None:
        labels = sorted(list(set(y_true).union(set(y_pred))))
    
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    K = len(labels)
    cm = np.zeros((K, K), dtype=np.int64)

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1

    return cm, labels


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    class_names_map: Optional[Dict[int, str]] = None
) -> Dict:
    """
    Calcula Precision, Recall, F1-Score y Soporte por clase,
    asi como los promedios Macro y Ponderado (Weighted).
    """
    cm, label_list = confusion_matrix_manual(y_true, y_pred, labels)
    K = len(label_list)
    total_samples = len(y_true)

    report = {"classes": {}, "macro_avg": {}, "weighted_avg": {}, "accuracy": accuracy_score_manual(y_true, y_pred)}

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, lbl in enumerate(label_list):
        name = class_names_map.get(lbl, f"Clase {lbl}") if class_names_map else f"Clase {lbl}"
        
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        report["classes"][lbl] = {
            "name": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

    # Macro Average (promedio simple no ponderado)
    report["macro_avg"] = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "support": int(np.sum(supports)),
    }

    # Weighted Average (promedio ponderado por soporte de cada clase)
    total_supp = np.sum(supports)
    if total_supp > 0:
        weights = np.array(supports) / total_supp
        report["weighted_avg"] = {
            "precision": float(np.sum(np.array(precisions) * weights)),
            "recall": float(np.sum(np.array(recalls) * weights)),
            "f1": float(np.sum(np.array(f1s) * weights)),
            "support": int(total_supp),
        }

    return report


def print_classification_report_table(report: Dict):
    """Imprime el reporte de clasificacion formateado en una tabla limpia de texto."""
    print("\n" + "=" * 82)
    print(" REPORTE DE CLASIFICACION (CALCULADO MANUALMENTE)")
    print("=" * 82)
    header = f"{'Movimiento':<42} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Soporte':>7}"
    print(header)
    print("-" * 82)

    for lbl, stats in report["classes"].items():
        name_str = f"[{lbl:02d}] {stats['name'][:35]}"
        print(f"{name_str:<42} {stats['precision']:>10.4f} {stats['recall']:>10.4f} {stats['f1']:>10.4f} {stats['support']:>7d}")

    print("-" * 82)
    acc = report["accuracy"]
    tot_supp = report["weighted_avg"]["support"]
    print(f"{'Exactitud Global (Accuracy)':<42} {' ':>10} {' ':>10} {acc:>10.4f} {tot_supp:>7d}")

    mac = report["macro_avg"]
    print(f"{'Macro Promedio (Macro Avg)':<42} {mac['precision']:>10.4f} {mac['recall']:>10.4f} {mac['f1']:>10.4f} {mac['support']:>7d}")

    wgt = report["weighted_avg"]
    print(f"{'Promedio Ponderado (Weighted Avg)':<42} {wgt['precision']:>10.4f} {wgt['recall']:>10.4f} {wgt['f1']:>10.4f} {wgt['support']:>7d}")
    print("=" * 82)


def print_confusion_matrix_ascii(cm: np.ndarray, labels: List[int], class_names_map: Optional[Dict[int, str]] = None):
    """Imprime la matriz de confusion en formato ASCII."""
    print("\n" + "=" * 90)
    print(" MATRIZ DE CONFUSION (Filas = Valor Real | Columnas = Prediccion)")
    print("=" * 90)
    
    # Encabezado de columnas
    col_hdr = "Real \\ Pred | " + " ".join([f"{lbl:>3}" for lbl in labels]) + " | Total"
    print(col_hdr)
    print("-" * len(col_hdr))

    for i, lbl in enumerate(labels):
        row_vals = " ".join([f"{cm[i, j]:>3}" for j in range(len(labels))])
        row_tot = np.sum(cm[i, :])
        print(f"  [{lbl:02d}]       | {row_vals} | {row_tot:>5}")

    print("=" * 90)


# ==============================================================================
# 5. DEMOSTRACION DE PREDICCIONES PASO A PASO
# ==============================================================================

def demonstrate_predictions(
    model: KNNClassifierManual,
    scaler: StandardScalerManual,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    idx_test: np.ndarray,
    feature_names: List[str],
    num_samples: int = 5,
    seed: int = 42
):
    """
    Ejecuta y despliega predicciones detalladas mostrando el vecindario paso a paso.
    """
    rng = np.random.default_rng(seed)
    total_test = len(y_test)
    sample_indices = rng.choice(total_test, size=min(num_samples, total_test), replace=False)

    print("\n" + "#" * 80)
    print(f" DEMOSTRACION DE PREDICCIONES EN CONSOLA ({num_samples} muestras seleccionadas)")
    print("#" * 80)

    for step, s_idx in enumerate(sample_indices, 1):
        x_scaled = X_test_scaled[s_idx]
        y_real = int(y_test[s_idx])
        original_sample_id = int(idx_test[s_idx])
        real_name = MOVEMENT_NAMES.get(y_real, f"Movimiento {y_real}")

        pred_class, neighbor_idx, neighbor_dists, neighbor_labels, class_probs = model.predict_single(x_scaled)
        pred_name = MOVEMENT_NAMES.get(pred_class, f"Movimiento {pred_class}")
        confidence = class_probs.get(pred_class, 0.0) * 100.0

        is_correct = (pred_class == y_real)
        status_tag = "[OK - CORRECTO]" if is_correct else "[X - ERROR]"

        print(f"\n--- Muestra de Prueba #{step} (ID original en dataset: {original_sample_id}) ---")
        print(f"  * Clase Real:       [{y_real:02d}] {real_name}")
        print(f"  * Clase Predicha:   [{pred_class:02d}] {pred_name} (Confianza: {confidence:.1f}%)")
        print(f"  * Resultado:        {status_tag}")
        print(f"  * Desglose de los {model.k} Vecinos Mas Cercanos (k-NN):")

        for rank, (n_i, n_d, n_l) in enumerate(zip(neighbor_idx, neighbor_dists, neighbor_labels), 1):
            n_name = MOVEMENT_NAMES.get(int(n_l), f"Mov {n_l}")
            match_flag = "✓" if n_l == y_real else " "
            print(f"      Vecino {rank}: ID_train={n_i:<4} | Dist={n_d:<7.4f} | Clase=[{n_l:02d}] {n_name:<40} {match_flag}")

    print("\n" + "#" * 80)


# ==============================================================================
# 6. PIPELINE COMPLETO DE ENTRENAMIENTO Y EVALUACION
# ==============================================================================

def run_experiment(
    csv_path: str = "data/rehab_tabular_features.csv",
    k: int = 5,
    metric: str = "euclidean",
    weights: str = "distance",
    test_size: float = 0.2,
    random_state: int = 42,
    num_demo_samples: int = 6
):
    """Ejecuta el ciclo completo de carga, entrenamiento manual, evaluacion y predicciones."""
    print("=" * 80)
    print(" EXPERIMENTO CLASIFICACION REHAB CON k-NN DESDE CERO (SIN LIBRERIAS ML)")
    print("=" * 80)
    print(f"Dataset CSV:     {csv_path}")
    print(f"Hiperparametros: k = {k}, Metrica = {metric}, Pesos = {weights}")
    print(f"Particion:       {(1-test_size)*100:.0f}% Train / {test_size*100:.0f}% Test (Estratificado)")
    print("=" * 80)

    # 1. Asegurar existencia de datos
    if not Path(csv_path).exists():
        print(f"No se encontro {csv_path}. Ejecutando extraccion ETL automatica...")
        generate_tabular_dataset(data_dir="data", output_csv_path=csv_path, verbose=True)

    # 2. Carga de datos
    print("\n[1/4] Cargando dataset tabular desde CSV...")
    X, y, feature_names, sample_ids, movement_names = load_tabular_csv(csv_path)
    print(f"      Registros cargados: {X.shape[0]} muestras, {X.shape[1]} caracteristicas.")

    # 3. Particion Train/Test
    print("\n[2/4] Dividiendo datos en Entrenamiento y Prueba (Manual y Estratificado)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split_manual(
        X, y, test_size=test_size, random_state=random_state, stratify=True
    )
    print(f"      Conjunto de Entrenamiento: {X_train.shape[0]} muestras")
    print(f"      Conjunto de Prueba:        {X_test.shape[0]} muestras")

    # 4. Estandarizacion Z-Score sin Data Leakage
    print("\n[3/4] Estandarizando variables con Z-Score Manual (ajustado en Train)...")
    scaler = StandardScalerManual()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"      Media de features escalados (Train): {np.mean(X_train_scaled):.4f}")
    print(f"      Std de features escalados   (Train): {np.std(X_train_scaled):.4f}")

    # 5. Ajuste y Prediccion k-NN
    print(f"\n[4/4] Entrenando e Infiriendo con k-NN Manual (k={k}, metric={metric})...")
    knn = KNNClassifierManual(k=k, metric=metric, weights=weights)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score_manual(y_test, y_pred)
    print(f"      Prediccion completada sobre {len(y_test)} muestras de prueba.")
    print(f"      >> EXACTITUD GLOBAL (ACCURACY): {acc * 100.0:.2f}% <<")

    # 6. Reporte y Matriz
    unique_labels = sorted(list(np.unique(y)))
    report = compute_classification_report(y_test, y_pred, labels=unique_labels, class_names_map=MOVEMENT_NAMES)
    print_classification_report_table(report)

    cm, lbls = confusion_matrix_manual(y_test, y_pred, labels=unique_labels)
    print_confusion_matrix_ascii(cm, lbls, MOVEMENT_NAMES)

    # 7. Demostracion de Predicciones
    demonstrate_predictions(
        model=knn,
        scaler=scaler,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        idx_test=idx_test,
        feature_names=feature_names,
        num_samples=num_demo_samples,
        seed=random_state
    )

    return knn, scaler, report


# ==============================================================================
# 7. INTERFAZ DE LINEA DE COMANDOS (CLI)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clasificador k-NN desde cero para el dataset REHAB (biomecanica de rehabilitacion)."
    )
    parser.add_argument("--csv", type=str, default="data/rehab_tabular_features.csv", help="Ruta al archivo CSV con las caracteristicas.")
    parser.add_argument("-k", "--neighbors", type=int, default=5, help="Numero de vecinos mas cercanos (k). Default: 5")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "manhattan"], help="Metrica de distancia.")
    parser.add_argument("--weights", type=str, default="distance", choices=["uniform", "distance"], help="Ponderacion de votos.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraccion para el conjunto de prueba (0.0 a 1.0). Default: 0.2")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad. Default: 42")
    parser.add_argument("--demo-samples", type=int, default=6, help="Numero de muestras de prueba a explicar en consola.")

    args = parser.parse_args()

    run_experiment(
        csv_path=args.csv,
        k=args.neighbors,
        metric=args.metric,
        weights=args.weights,
        test_size=args.test_size,
        random_state=args.seed,
        num_demo_samples=args.demo_samples
    )


if __name__ == "__main__":
    main()
