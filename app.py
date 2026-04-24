from flask import Flask, render_template, jsonify
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    load_digits,
    load_diabetes,
    fetch_california_housing,
    make_classification,
    make_regression
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import xgboost as xgb
import numpy as np

app = Flask(__name__)


MODELOS = {
    "cancer": {
        "nombre": "Cáncer de Mama",
        "descripcion": "Clasificación binaria de tumores benignos y malignos.",
        "tipo": "clasificacion",
        "dataset": "breast_cancer"
    },
    "iris": {
        "nombre": "Flores Iris",
        "descripcion": "Clasificación multiclase de especies de flores.",
        "tipo": "clasificacion",
        "dataset": "iris"
    },
    "wine": {
        "nombre": "Tipos de Vino",
        "descripcion": "Clasificación multiclase de vinos.",
        "tipo": "clasificacion",
        "dataset": "wine"
    },
    "digits": {
        "nombre": "Dígitos Escritos",
        "descripcion": "Clasificación de números escritos a mano.",
        "tipo": "clasificacion",
        "dataset": "digits"
    },
    "sintetica_clasificacion": {
        "nombre": "Clasificación Sintética",
        "descripcion": "Datos generados artificialmente para clasificación.",
        "tipo": "clasificacion",
        "dataset": "synthetic_classification"
    },
    "diabetes": {
        "nombre": "Diabetes",
        "descripcion": "Regresión para predecir progresión de enfermedad.",
        "tipo": "regresion",
        "dataset": "diabetes"
    },
    "california": {
        "nombre": "California Housing",
        "descripcion": "Regresión para predecir precios de viviendas.",
        "tipo": "regresion",
        "dataset": "california"
    },
    "sintetica_regresion": {
        "nombre": "Regresión Sintética",
        "descripcion": "Datos generados artificialmente para predicción numérica.",
        "tipo": "regresion",
        "dataset": "synthetic_regression"
    }
}


def cargar_dataset(nombre):
    if nombre == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, data.feature_names, data.target_names

    if nombre == "iris":
        data = load_iris()
        return data.data, data.target, data.feature_names, data.target_names

    if nombre == "wine":
        data = load_wine()
        return data.data, data.target, data.feature_names, data.target_names

    if nombre == "digits":
        data = load_digits()
        feature_names = [f"pixel_{i}" for i in range(data.data.shape[1])]
        target_names = [str(i) for i in data.target_names]
        return data.data, data.target, feature_names, target_names

    if nombre == "diabetes":
        data = load_diabetes()
        return data.data, data.target, data.feature_names, ["valor"]

    if nombre == "california":
        data = fetch_california_housing()
        return data.data, data.target, data.feature_names, ["precio"]

    if nombre == "synthetic_classification":
        X, y = make_classification(
            n_samples=1200,
            n_features=12,
            n_informative=8,
            n_redundant=2,
            n_classes=3,
            random_state=42
        )
        feature_names = [f"variable_{i+1}" for i in range(X.shape[1])]
        target_names = ["Clase A", "Clase B", "Clase C"]
        return X, y, feature_names, target_names

    if nombre == "synthetic_regression":
        X, y = make_regression(
            n_samples=1200,
            n_features=10,
            noise=18,
            random_state=42
        )
        feature_names = [f"variable_{i+1}" for i in range(X.shape[1])]
        return X, y, feature_names, ["valor"]

    data = load_iris()
    return data.data, data.target, data.feature_names, data.target_names


def top_importancias(modelo, feature_names):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[-10:][::-1]

    features = []
    valores = []

    for i in indices:
        features.append(str(feature_names[i]))
        valores.append(round(float(importancias[i]), 4))

    return features, valores


def entrenar_clasificacion(dataset_name):
    X, y, feature_names, target_names = cargar_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    modelo = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        eval_metric="mlogloss",
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    matriz = confusion_matrix(y_test, y_pred)

    features, importancias = top_importancias(modelo, feature_names)

    clases = [str(c) for c in target_names]

    conteo_predicciones = []
    for i in range(len(clases)):
        conteo_predicciones.append(int(np.sum(y_pred == i)))

    return {
        "tipo": "clasificacion",
        "accuracy": round(float(accuracy) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "matriz": matriz.tolist(),
        "features": features,
        "importancias": importancias,
        "clases": clases,
        "conteo_predicciones": conteo_predicciones,
        "total_entrenamiento": len(X_train),
        "total_prueba": len(X_test),
        "primeras_probabilidades": np.round(y_proba[:5], 3).tolist()
    }


def entrenar_regresion(dataset_name):
    X, y, feature_names, target_names = cargar_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    modelo = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        objective="reg:squarederror",
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    features, importancias = top_importancias(modelo, feature_names)

    reales = np.round(y_test[:20], 2).tolist()
    predichos = np.round(y_pred[:20], 2).tolist()

    return {
        "tipo": "regresion",
        "mae": round(float(mae), 3),
        "mse": round(float(mse), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2) * 100, 2),
        "features": features,
        "importancias": importancias,
        "total_entrenamiento": len(X_train),
        "total_prueba": len(X_test),
        "reales": reales,
        "predichos": predichos
    }


@app.route("/")
def index():
    return render_template("index.html", modelos=MODELOS)


@app.route("/modelo/<modelo_id>")
def ejecutar_modelo(modelo_id):
    if modelo_id not in MODELOS:
        return jsonify({"error": "Modelo no encontrado"}), 404

    info = MODELOS[modelo_id]

    try:
        if info["tipo"] == "clasificacion":
            resultados = entrenar_clasificacion(info["dataset"])
        else:
            resultados = entrenar_regresion(info["dataset"])

        return jsonify({
            "nombre": info["nombre"],
            "descripcion": info["descripcion"],
            "tipo": info["tipo"],
            "resultados": resultados
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)