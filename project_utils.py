from pathlib import Path
import json
import re
import zipfile
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42
FP_COST = 10
FN_COST = 500


def crear_carpetas(base_path="."):
    base = Path(base_path)
    for carpeta in [
        "data/raw",
        "data/processed",
        "artifacts/metrics",
        "artifacts/models",
        "artifacts/figures",
        "artifacts/tables",
    ]:
        (base / carpeta).mkdir(parents=True, exist_ok=True)


def leer_csv_aps(path):
    path = Path(path)
    skip_rows = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith("class,"):
                skip_rows = i
                break
    return pd.read_csv(path, skiprows=skip_rows, na_values="na")


def cargar_o_descargar_dataset(raw_dir="data/raw"):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_file = raw_dir / "aps_failure_training_set.csv"
    test_file = raw_dir / "aps_failure_test_set.csv"
    if train_file.exists():
        train_df = leer_csv_aps(train_file)
        test_df = None
        if test_file.exists():
            test_df = leer_csv_aps(test_file)
        return train_df, test_df

    zip_file = raw_dir / "aps_failure_at_scania_trucks.zip"
    url = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"

    if not zip_file.exists():
        urlretrieve(url, zip_file)

    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(raw_dir)

    train_df = leer_csv_aps(train_file)
    test_df = None
    if test_file.exists():
        test_df = leer_csv_aps(test_file)
    return train_df, test_df


def limpiar_nombres_columnas(df):
    df = df.copy()
    cols = []
    for col in df.columns:
        col = str(col).strip().lower()
        col = re.sub(r"[^0-9a-zA-Z_]+", "_", col)
        col = re.sub(r"_+", "_", col).strip("_")
        cols.append(col)
    df.columns = cols
    return df


def convertir_faltantes(df):
    df = df.copy()
    df = df.replace(["na", "NA", "?", "", "null", "None"], np.nan)
    for col in df.columns:
        if col != "class":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def separar_x_y(df, target="class"):
    X = df.drop(columns=[target])
    y = df[target]
    if y.dtype == object:
        y = y.astype(str).str.lower().map({"neg": 0, "pos": 1})
    return X, y.astype(int)


def crear_train_test(df, target="class", test_size=0.2, random_state=RANDOM_STATE):
    X, y = separar_x_y(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train = X_train.copy()
    train[target] = y_train.values
    test = X_test.copy()
    test[target] = y_test.values
    return train, test


def guardar_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def cargar_csv(path):
    return pd.read_csv(path)


def guardar_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def guardar_modelo(modelo, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, path)


def cargar_modelo(path):
    return joblib.load(path)


def calcular_matriz_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def calcular_coste_errores(y_true, y_pred, fp_cost=FP_COST, fn_cost=FN_COST):
    tn, fp, fn, tp = calcular_matriz_confusion(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "coste": int(fp * fp_cost + fn * fn_cost)}


def obtener_scores(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X)
    return None


def calcular_metricas(y_true, y_pred, y_score=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if y_score is None:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    else:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            out["roc_auc"] = np.nan
        try:
            out["pr_auc"] = average_precision_score(y_true, y_score)
        except ValueError:
            out["pr_auc"] = np.nan
    out.update(calcular_coste_errores(y_true, y_pred))
    return out


def guardar_figura(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def obtener_variables_numericas(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def evaluar_modelo(nombre, modelo, X, y):
    y_pred = modelo.predict(X)
    y_score = obtener_scores(modelo, X)
    out = calcular_metricas(y, y_pred, y_score)
    out["modelo"] = nombre
    return out


def crear_resumen_resultados(filas):
    df = pd.DataFrame(filas)
    return df[["modelo"] + [c for c in df.columns if c != "modelo"]]


def classification_report_dataframe(y_true, y_pred):
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T