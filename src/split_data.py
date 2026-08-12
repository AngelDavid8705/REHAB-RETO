"""Split train/validation/test (estratificado) del dataset consolidado de REHAB."""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def split_train_val_test(X, y, val_size=0.15, test_size=0.15, random_state=RANDOM_STATE):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def resumen_por_clase(y, nombre):
    clases, conteos = np.unique(y, return_counts=True)
    print(f"{nombre}: {len(y)} muestras")
    for c, n in zip(clases, conteos):
        print(f"  clase {c:02d}: {n} ({100 * n / len(y):.1f}%)")


if __name__ == "__main__":
    import sys

    npz_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/rehab_exercise_consolidado.npz"
    data = np.load(npz_path)
    X, y = data["X"], data["y"]

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)

    print(f"Total: {len(y)} muestras\n")
    resumen_por_clase(y_train, "Train")
    print()
    resumen_por_clase(y_val, "Validation")
    print()
    resumen_por_clase(y_test, "Test")

    out_path = Path(npz_path).with_name("rehab_exercise_split.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print(f"\nGuardado en {out_path}")
