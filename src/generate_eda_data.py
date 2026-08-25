"""
src/generate_eda_data.py
Script de analisis exploratorio de datos (EDA) exhaustivo para el dataset REHAB (Rehab_exercise).
Enfocado exclusivamente en el conjunto de entrenamiento (X_train, y_train).
Genera figuras profesionales de alta resolucion y tablas estadisticas para el reporte PDF.
"""

import os
import sys
import json
from pathlib import Path

# Agregar raíz del proyecto a sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configuracion de estilo visual profesional
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EBEBEB'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CHANNEL_NAMES = [
    "IMU1_Pitch", "IMU1_Yaw", "IMU1_Roll",
    "IMU2_Pitch", "IMU2_Yaw", "IMU2_Roll",
    "Glove_Thumb", "Glove_Index", "Glove_Middle",
    "Glove_Ring", "Glove_Pinky", "Glove_Pitch"
]

CHANNEL_DESCRIPTIONS = [
    "IMU 1 (Brazo/Muslo) - Pitch (°)",
    "IMU 1 (Brazo/Muslo) - Yaw (°)",
    "IMU 1 (Brazo/Muslo) - Roll (°)",
    "IMU 2 (Antebrazo/Pantorrilla) - Pitch (°)",
    "IMU 2 (Antebrazo/Pantorrilla) - Yaw (°)",
    "IMU 2 (Antebrazo/Pantorrilla) - Roll (°)",
    "Guante Flex - Pulgar F1 (u.a.)",
    "Guante Flex - Indice F2 (u.a.)",
    "Guante Flex - Medio F3 (u.a.)",
    "Guante Flex - Anular F4 (u.a.)",
    "Guante Flex - Menique F5 (u.a.)",
    "Guante IMU - Pitch Muneca S5 (°)"
]

MOVEMENT_NAMES = {
    0: "Bobath Handshake",
    1: "Bobath Flexion/Extension",
    2: "Bobath Forward Flexion/Ext.",
    3: "Bobath Ant./Post. Rotation",
    4: "Elbow Flex. & Wrist Comp.",
    5: "Wrist Flexion & Extension",
    6: "Finger-to-Finger Training",
    7: "Ball Gripping",
    8: "Shoulder Int./Ext. Rotation",
    9: "Breast Expansion",
    10: "Flex.-Press. Rot. Fwd/Back",
    11: "Elbow Joint Flex. & Touch",
    12: "Shoulder Touch Training",
    13: "Ankle Ext. & Knee Rot.",
    14: "Knee Flexion & Extension (Corrupto)",
    15: "Hip Flexion & Extension"
}

PALETTE_15 = sns.color_palette("tab20", 15)


def load_and_split_data(data_dir="data"):
    from src.data_loader import build_dataset
    X, y, meta = build_dataset(data_dir, incluir_mov14_parcial=False, verbose=False)
    
    # Split 70% Train, 15% Val, 15% Test estratificado
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    val_ratio = 0.15 / (1.0 - 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=RANDOM_STATE
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, X, y, meta


def compute_step1_overview_stats(X_train, y_train, X_full, y_full):
    N_train, T, C = X_train.shape
    total_points = N_train * T * C
    mem_mb = X_train.nbytes / (1024 * 1024)
    
    nan_count = int(np.isnan(X_train).sum())
    inf_count = int(np.isinf(X_train).sum())
    zero_count = int((X_train == 0.0).sum())
    zero_pct = (zero_count / total_points) * 100.0
    
    # Conteo por clase
    classes, counts = np.unique(y_train, return_counts=True)
    class_df = pd.DataFrame({
        "class_id": classes,
        "movement_name": [MOVEMENT_NAMES[c] for c in classes],
        "train_samples": counts,
        "train_pct": (counts / N_train) * 100.0,
        "full_samples": [int((y_full == c).sum()) for c in classes]
    })
    
    # Resumen por canal
    channel_stats = []
    for c_idx in range(C):
        flat_c = X_train[:, :, c_idx].flatten()
        q25, q50, q75 = np.percentile(flat_c, [25, 50, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers_count = np.sum((flat_c < lower_bound) | (flat_c > upper_bound))
        outliers_pct = (outliers_count / len(flat_c)) * 100.0
        
        channel_stats.append({
            "channel_idx": c_idx,
            "channel_name": CHANNEL_NAMES[c_idx],
            "sensor_type": "IMU Euler" if c_idx < 6 else ("Flex Sensor" if c_idx < 11 else "Glove IMU"),
            "mean": float(np.mean(flat_c)),
            "std": float(np.std(flat_c)),
            "median": float(q50),
            "iqr": float(iqr),
            "min": float(np.min(flat_c)),
            "max": float(np.max(flat_c)),
            "skewness": float(stats.skew(flat_c)),
            "kurtosis": float(stats.kurtosis(flat_c)),
            "outliers_pct": float(outliers_pct),
            "zero_pct": float((np.sum(flat_c == 0.0) / len(flat_c)) * 100.0)
        })
    
    channel_df = pd.DataFrame(channel_stats)
    
    summary = {
        "N_train": N_train,
        "time_steps": T,
        "channels": C,
        "total_data_points": total_points,
        "memory_mb": mem_mb,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_count": zero_count,
        "zero_pct": zero_pct,
        "classes_count": len(classes),
        "min_class_samples": int(counts.min()),
        "max_class_samples": int(counts.max()),
        "imbalance_ratio": float(counts.max() / counts.min()),
        "shannon_entropy": float(stats.entropy(counts / N_train))
    }
    
    return summary, class_df, channel_df


def generate_figures(X_train, y_train, class_df, channel_df, out_dir="reports/figures"):
    os.makedirs(out_dir, exist_ok=True)
    N_train, T, C = X_train.shape
    
    # ----------------------------------------------------
    # FIG 1: Distribucion de clases en Train y Balance
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    bars = ax.bar(
        [f"M{c:02d}\n{MOVEMENT_NAMES[c][:13]}" for c in class_df["class_id"]],
        class_df["train_samples"],
        color="#2B5B84",
        edgecolor="#1A365D",
        linewidth=0.8,
        width=0.65
    )
    # Highlight min and max
    min_idx = class_df["train_samples"].idxmin()
    max_idx = class_df["train_samples"].idxmax()
    bars[min_idx].set_color("#C53030")
    bars[max_idx].set_color("#2F855A")
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#2D3748")
                    
    mean_reps = class_df["train_samples"].mean()
    ax.axhline(mean_reps, color="#DD6B20", linestyle="--", linewidth=1.5,
               label=f"Media por clase = {mean_reps:.1f} repeticiones")
               
    ax.set_title("Figura 1: Distribución y Balance de Clases en el Set de Entrenamiento (N=2,979)",
                 fontsize=13, fontweight='bold', pad=15, color="#1A202C")
    ax.set_xlabel("Movimiento de Rehabilitación (Clase)", fontsize=10, labelpad=8, fontweight='bold')
    ax.set_ylabel("Número de Muestras (Trials)", fontsize=10, labelpad=8, fontweight='bold')
    ax.set_ylim(0, class_df["train_samples"].max() * 1.15)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0")
    plt.xticks(rotation=40, ha="right", fontsize=8.5)
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "fig1_dataset_overview_classes.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # FIG 2: Distribuciones y Boxplots por Canal (12 Canales)
    # ----------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=300)
    axes = axes.flatten()
    
    for c_idx in range(C):
        ax = axes[c_idx]
        flat_c = X_train[:, :, c_idx].flatten()
        # Muestra aleatoria de 50k puntos para kde/histogram rapido
        sample_pts = np.random.choice(flat_c, size=min(50000, len(flat_c)), replace=False)
        
        color = "#3182CE" if c_idx < 6 else ("#38A169" if c_idx < 11 else "#805AD5")
        ax.hist(sample_pts, bins=80, density=True, alpha=0.65, color=color, edgecolor='none')
        
        # Superponer boxplot en miniatura
        ax2 = ax.twinx()
        ax2.boxplot(sample_pts, orientation='horizontal', positions=[0.5], widths=[0.3],
                    patch_artist=True, boxprops=dict(facecolor=color, alpha=0.3),
                    flierprops=dict(marker='.', markersize=2, alpha=0.2, color='gray'),
                    medianprops=dict(color='#E53E3E', linewidth=1.5))
        ax2.set_ylim(0, 2)
        ax2.set_yticks([])
        
        q50 = np.median(flat_c)
        skew = stats.skew(flat_c)
        kurt = stats.kurtosis(flat_c)
        
        ax.set_title(f"{CHANNEL_NAMES[c_idx]}\n[Median={q50:.2f}, Skew={skew:.2f}, Kurt={kurt:.2f}]",
                     fontsize=9, fontweight='bold', color="#2D3748")
        ax.tick_params(axis='both', labelsize=7.5)
        ax.set_ylabel("Densidad", fontsize=7.5)
        
    fig.suptitle("Figura 2: Análisis Univariado de Distribuciones, Asimetría y Outliers (12 Canales Cinemáticos)",
                 fontsize=14, fontweight='bold', y=0.99, color="#1A202C")
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "fig2_channel_distributions_box_violin.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # FIG 3: Perfiles Cinemáticos Promedio (Trayectorias Temporales)
    # ----------------------------------------------------
    # Seleccionamos 4 movimientos arquetipicos: 0 (Bobath), 5 (Wrist), 7 (Ball grip), 15 (Hip)
    selected_movs = [0, 5, 7, 15]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=300)
    axes = axes.flatten()
    time_axis = np.arange(T)
    
    for idx, mov_id in enumerate(selected_movs):
        ax = axes[idx]
        mask = (y_train == mov_id)
        X_mov = X_train[mask]  # (N_mov, 880, 12)
        
        # Graficamos 4 canales clave: IMU1_Pitch (0), IMU2_Pitch (3), Glove_Thumb (6), Glove_Index (7)
        channels_to_plot = [(0, "IMU1 Pitch", "#3182CE", "-"),
                            (3, "IMU2 Pitch", "#E53E3E", "--"),
                            (6, "Glove Thumb", "#38A169", "-."),
                            (7, "Glove Index", "#D69E2E", ":")]
                            
        for ch, ch_lbl, ch_color, ch_ls in channels_to_plot:
            mean_traj = np.mean(X_mov[:, :, ch], axis=0)
            std_traj = np.std(X_mov[:, :, ch], axis=0)
            ax.plot(time_axis, mean_traj, label=ch_lbl, color=ch_color, linestyle=ch_ls, linewidth=1.5)
            ax.fill_between(time_axis, mean_traj - std_traj, mean_traj + std_traj, color=ch_color, alpha=0.15)
            
        ax.set_title(f"Mov {mov_id:02d}: {MOVEMENT_NAMES[mov_id]} (n={mask.sum()} trials)",
                     fontsize=11, fontweight='bold', color="#2D3748")
        ax.set_xlabel("Puntos de Tiempo (880 timesteps @ señal)", fontsize=9)
        ax.set_ylabel("Amplitud Normalizada (Media ± 1 DE)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")
        ax.grid(True, linestyle="--", alpha=0.6)
        
    fig.suptitle("Figura 3: Perfiles Cinemáticos Promedio y Variabilidad Inter-Sujeto (Banda ±1 DE)",
                 fontsize=14, fontweight='bold', y=0.99, color="#1A202C")
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "fig3_temporal_kinematic_trajectories.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # FIG 4: Matrices de Correlacion Pearson y Spearman
    # ----------------------------------------------------
    # Aplanar por canal para correlacion global
    X_flat_channels = X_train.reshape(-1, C)
    # Tomar submuestra representativa de 100,000 puntos para estabilidad numerica
    sample_corr_idx = np.random.choice(len(X_flat_channels), size=100000, replace=False)
    sub_sample_flat = X_flat_channels[sample_corr_idx]
    
    corr_pearson = np.corrcoef(sub_sample_flat.T)
    corr_spearman, _ = stats.spearmanr(sub_sample_flat)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    mask_tri = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
    
    sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES, ax=ax1,
                cbar_kws={'label': 'Coeficiente de Pearson (r)', 'shrink': 0.8},
                annot_kws={'size': 7.5, 'weight': 'bold'})
    ax1.set_title("Matriz de Correlación Lineal de Pearson", fontsize=12, fontweight='bold', pad=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', rotation=0, labelsize=8)
    
    sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES, ax=ax2,
                cbar_kws={'label': 'Coeficiente de Rango de Spearman (ρ)', 'shrink': 0.8},
                annot_kws={'size': 7.5, 'weight': 'bold'})
    ax2.set_title("Matriz de Correlación Monotónica de Spearman", fontsize=12, fontweight='bold', pad=12)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.tick_params(axis='y', rotation=0, labelsize=8)
    
    fig.suptitle("Figura 4: Análisis Multivariado de Colinealidad y Redundancia entre Sensores",
                 fontsize=14, fontweight='bold', y=0.99, color="#1A202C")
    plt.tight_layout()
    fig4_path = os.path.join(out_dir, "fig4_correlation_heatmaps.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # FIG 5: Interacciones Cruzadas entre Canales Clave (Scatter 2D)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    # Interaccion 1: Glove Ring (ch 9) vs Glove Middle (ch 8) - Redundancia de dedos
    ax = axes[0]
    ax.scatter(sub_sample_flat[:4000, 8], sub_sample_flat[:4000, 9],
               alpha=0.25, color="#38A169", s=8, edgecolors='none')
    r_val = corr_pearson[8, 9]
    ax.set_title(f"Glove Middle vs Glove Ring\n(Alta Colinealidad: r={r_val:.2f})", fontsize=10, fontweight='bold')
    ax.set_xlabel("Flexión Dedo Medio (u.a.)", fontsize=8.5)
    ax.set_ylabel("Flexión Dedo Anular (u.a.)", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Interaccion 2: IMU1 Pitch (0) vs IMU2 Pitch (3) - Coordinacion cinemática brazo/antebrazo
    ax = axes[1]
    ax.scatter(sub_sample_flat[:4000, 0], sub_sample_flat[:4000, 3],
               alpha=0.25, color="#3182CE", s=8, edgecolors='none')
    r_val = corr_pearson[0, 3]
    ax.set_title(f"IMU1 Pitch vs IMU2 Pitch\n(Coordinación Brazo/Antebrazo: r={r_val:.2f})", fontsize=10, fontweight='bold')
    ax.set_xlabel("IMU 1 Pitch (°)", fontsize=8.5)
    ax.set_ylabel("IMU 2 Pitch (°)", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Interaccion 3: Glove Thumb (6) vs IMU1 Roll (2) - Desacoplamiento distal/proximal
    ax = axes[2]
    ax.scatter(sub_sample_flat[:4000, 6], sub_sample_flat[:4000, 2],
               alpha=0.25, color="#805AD5", s=8, edgecolors='none')
    r_val = corr_pearson[6, 2]
    ax.set_title(f"Glove Thumb vs IMU1 Roll\n(Desacoplamiento Distal/Proximal: r={r_val:.2f})", fontsize=10, fontweight='bold')
    ax.set_xlabel("Guante Pulgar (u.a.)", fontsize=8.5)
    ax.set_ylabel("IMU 1 Roll (°)", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig.suptitle("Figura 5: Dispersión e Interacciones Cinemáticas Clave entre Canales de Sensores",
                 fontsize=14, fontweight='bold', y=0.99, color="#1A202C")
    plt.tight_layout()
    fig5_path = os.path.join(out_dir, "fig5_sensor_interactions_scatter.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # FIG 6: Analisis en Dominio de Frecuencia (PSD / Welch)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    # Subplot 1: PSD por Canal para Mov 07 (Ball Gripping)
    ax = axes[0]
    mask_07 = (y_train == 7)
    X_07 = X_train[mask_07]
    for ch_idx, ch_name in enumerate(["IMU1_Pitch", "IMU2_Pitch", "Glove_Thumb", "Glove_Index"]):
        real_idx = [0, 3, 6, 7][ch_idx]
        sig = X_07[:, :, real_idx]
        freqs, psd = welch(sig, fs=50.0, nperseg=256, axis=1)  # asumiendo fs ~ 50Hz tipico en IMUs REHAB
        mean_psd = np.mean(psd, axis=0)
        ax.semilogy(freqs, mean_psd, label=ch_name, linewidth=1.5)
        
    ax.set_title("Espectro de Potencia (PSD) - Mov 07 (Ball Gripping)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Frecuencia (Hz)", fontsize=9)
    ax.set_ylabel("Densidad Espectral de Potencia (V²/Hz)", fontsize=9)
    ax.set_xlim(0, 20)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Subplot 2: Comparacion de Energia Espectral entre 4 Movimientos en IMU1 Pitch
    ax = axes[1]
    for mov_id in [0, 5, 7, 15]:
        mask = (y_train == mov_id)
        sig = X_train[mask, :, 0]
        freqs, psd = welch(sig, fs=50.0, nperseg=256, axis=1)
        mean_psd = np.mean(psd, axis=0)
        ax.semilogy(freqs, mean_psd, label=f"Mov {mov_id:02d}: {MOVEMENT_NAMES[mov_id][:15]}", linewidth=1.5)
        
    ax.set_title("Comparación Espectral Inter-Clase (Canal IMU1_Pitch)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Frecuencia (Hz)", fontsize=9)
    ax.set_ylabel("Densidad Espectral de Potencia (V²/Hz)", fontsize=9)
    ax.set_xlim(0, 20)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    fig.suptitle("Figura 6: Análisis Espectral (FFT / Welch PSD) y Dinámica de Frecuencia del Movimiento",
                 fontsize=14, fontweight='bold', y=0.99, color="#1A202C")
    plt.tight_layout()
    fig6_path = os.path.join(out_dir, "fig6_spectral_fft_analysis.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # FIG 7: Dashboard de Calidad de Datos, Outliers y Diagrama Forense
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)
    
    # Grafica 1: Porcentaje de Outliers (IQR 1.5) y Zeros por Canal
    x_pos = np.arange(C)
    width = 0.35
    ax1.bar(x_pos - width/2, channel_df["outliers_pct"], width, label="Outliers (1.5*IQR)", color="#E53E3E", edgecolor="#9B2C2C")
    ax1.bar(x_pos + width/2, channel_df["zero_pct"], width, label="Valores Cero Exactos", color="#4A5568", edgecolor="#2D3748")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=8.5)
    ax1.set_ylabel("Porcentaje de Datos (%)", fontsize=9.5, fontweight='bold')
    ax1.set_title("Tasas de Outliers y Porcentaje de Ceros por Canal", fontsize=11, fontweight='bold')
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Grafica 2: Resumen Forense de Integridad del Dataset (16 Movimientos)
    file_status = []
    for m in range(16):
        if m == 14:
            file_status.append({"mov": m, "name": MOVEMENT_NAMES[m][:15], "imu_ok": False, "glove_ok": True, "usable": "Parcial (Solo Guante)"})
        else:
            file_status.append({"mov": m, "name": MOVEMENT_NAMES[m][:15], "imu_ok": True, "glove_ok": True, "usable": "Completo (12 Canales)"})
            
    df_forensic = pd.DataFrame(file_status)
    y_pos = np.arange(16)
    colors_status = ["#2F855A" if row["imu_ok"] else "#C53030" for _, row in df_forensic.iterrows()]
    
    ax2.barh(y_pos, [12 if row["imu_ok"] else 6 for _, row in df_forensic.iterrows()],
             color=colors_status, edgecolor="#1A202C", height=0.65)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"M{m:02d} ({MOVEMENT_NAMES[m][:13]})" for m in range(16)], fontsize=7.5)
    ax2.set_xlabel("Número de Canales Válidos Disponibles (Max=12)", fontsize=9.5, fontweight='bold')
    ax2.set_title("Auditoría de Integridad Multimodal (16 Movimientos)", fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 14)
    ax2.axvline(12, color="#2F855A", linestyle=":", alpha=0.7)
    
    # Anotacion en Mov 14
    ax2.annotate("ERROR ESTRUCTURAL FUENTE:\n014_1.npy corrupto (UTF-8 binary swap)",
                 xy=(6, 14), xytext=(8, 13.5),
                 arrowprops=dict(facecolor='#C53030', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8, fontweight='bold', color="#C53030",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFF5F5", ec="#FEB2B2"))
                 
    fig.suptitle("Figura 7: Diagnóstico de Calidad de Datos, Outliers y Auditoría Forense de Integridad",
                 fontsize=14, fontweight='bold', y=0.99, color="#1A202C")
    plt.tight_layout()
    fig7_path = os.path.join(out_dir, "fig7_data_quality_anomalies_outliers.png")
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # FIG 8: Espacio Latente PCA y t-SNE de Ensayos de Entrenamiento
    # ----------------------------------------------------
    # Extraer caracteristicas agregadas por trial (Media, DE, RMS, Min, Max por canal = 12 * 5 = 60 features)
    features_list = []
    for i in range(N_train):
        trial = X_train[i]  # (880, 12)
        feat = np.hstack([
            np.mean(trial, axis=0),
            np.std(trial, axis=0),
            np.sqrt(np.mean(trial**2, axis=0)),
            np.min(trial, axis=0),
            np.max(trial, axis=0)
        ])
        features_list.append(feat)
        
    X_feats = np.array(features_list)  # (2979, 60)
    
    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_feats)
    evr = pca.explained_variance_ratio_
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, max_iter=1000)
    X_tsne = tsne.fit_transform(X_feats)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap="tab20", s=15, alpha=0.75, edgecolors='none')
    ax1.set_title(f"Proyección 2D PCA (Varianza Explicada: PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%)",
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel("Componente Principal 1 (PC1)", fontsize=9)
    ax1.set_ylabel("Componente Principal 2 (PC2)", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap="tab20", s=15, alpha=0.75, edgecolors='none')
    ax2.set_title("Proyección no lineal 2D t-SNE (Separabilidad de Clases)",
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel("Dimensión t-SNE 1", fontsize=9)
    ax2.set_ylabel("Dimensión t-SNE 2", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], orientation='horizontal', fraction=0.06, pad=0.15)
    cbar.set_ticks(range(16))
    cbar.set_ticklabels([f"M{m:02d}" for m in range(16)], fontsize=8)
    cbar.set_label("Movimiento de Rehabilitación (Clases 00 a 15)", fontsize=10, fontweight='bold', labelpad=6)
    
    fig.suptitle("Figura 8: Estructura del Espacio Latente y Separabilidad Inter-Clase en Conjunto de Entrenamiento",
                 fontsize=14, fontweight='bold', y=0.98, color="#1A202C")
    plt.tight_layout()
    fig8_path = os.path.join(out_dir, "fig8_pca_tsne_latent_space.png")
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    
    print("Todas las figuras de alta resolucion generadas exitosamente en:", out_dir)
    return {
        "fig1": fig1_path,
        "fig2": fig2_path,
        "fig3": fig3_path,
        "fig4": fig4_path,
        "fig5": fig5_path,
        "fig6": fig6_path,
        "fig7": fig7_path,
        "fig8": fig8_path,
        "pca_evr": [float(evr[0]), float(evr[1])]
    }


def main():
    print("Iniciando procesamiento y analisis estadistico profundo...")
    X_train, y_train, X_val, y_val, X_test, y_test, X_full, y_full, meta = load_and_split_data("data")
    
    summary, class_df, channel_df = compute_step1_overview_stats(X_train, y_train, X_full, y_full)
    fig_paths = generate_figures(X_train, y_train, class_df, channel_df)
    
    # Guardar resultados en JSON para utilizarlos en ReportLab
    eda_data = {
        "summary": summary,
        "classes": class_df.to_dict(orient="records"),
        "channels": channel_df.to_dict(orient="records"),
        "figures": fig_paths
    }
    
    out_json = "reports/eda_results.json"
    with open(out_json, "w") as f:
        json.dump(eda_data, f, indent=2)
        
    print(f"Resultados estadisticos guardados en {out_json}")
    print("\n--- RESUMEN GENERAL (TRAIN SET) ---")
    print(f"Muestras de entrenamiento: {summary['N_train']:,}")
    print(f"Dimensiones de señal: {summary['time_steps']} timesteps x {summary['channels']} canales")
    print(f"Total puntos evaluados: {summary['total_data_points']:,}")
    print(f"NaNs: {summary['nan_count']} | Infs: {summary['inf_count']}")
    print(f"Clases validas: {summary['classes_count']} (Rango muestras: {summary['min_class_samples']} - {summary['max_class_samples']})")
    print(f"Imbalance Ratio (Max/Min): {summary['imbalance_ratio']:.2f}")


if __name__ == "__main__":
    main()
