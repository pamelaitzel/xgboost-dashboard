from flask import Flask, render_template, jsonify, request
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    load_digits,
    load_diabetes,
    fetch_california_housing,
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
    r2_score,
)
from sklearn.datasets import make_classification
import xgboost as xgb
import numpy as np
import os

app = Flask(__name__)

MODELOS = {
    "cancer": {
        "nombre": "Cáncer de Mama",
        "descripcion": "Clasificación de tumores benignos y malignos.",
        "tipo": "clasificacion",
        "dataset": "breast_cancer",
    },
    "iris": {
        "nombre": "Flores Iris",
        "descripcion": "Clasificación de especies de flores.",
        "tipo": "clasificacion",
        "dataset": "iris",
    },
    "wine": {
        "nombre": "Vinos",
        "descripcion": "Clasificación de tipos de vino.",
        "tipo": "clasificacion",
        "dataset": "wine",
    },
    "digits": {
        "nombre": "Dígitos",
        "descripcion": "Clasificación de números escritos a mano.",
        "tipo": "clasificacion",
        "dataset": "digits",
    },
    "diabetes": {
        "nombre": "Diabetes",
        "descripcion": "Regresión para predecir progresión de enfermedad.",
        "tipo": "regresion",
        "dataset": "diabetes",
    },
    "california": {
        "nombre": "California Housing",
        "descripcion": "Regresión para predecir precios de viviendas.",
        "tipo": "regresion",
        "dataset": "california",
    },
}


def cargar_dataset(nombre):
    if nombre == "breast_cancer":
        d = load_breast_cancer()
        return d.data, d.target, d.feature_names, d.target_names

    if nombre == "iris":
        d = load_iris()
        return d.data, d.target, d.feature_names, d.target_names

    if nombre == "wine":
        d = load_wine()
        return d.data, d.target, d.feature_names, d.target_names

    if nombre == "digits":
        d = load_digits()
        features = [f"pixel_{i}" for i in range(d.data.shape[1])]
        clases = [str(i) for i in d.target_names]
        return d.data, d.target, features, clases

    if nombre == "diabetes":
        d = load_diabetes()
        return d.data, d.target, d.feature_names, ["valor"]

    if nombre == "california":
        d = fetch_california_housing()
        return d.data, d.target, d.feature_names, ["precio"]

    raise ValueError("Dataset no encontrado")


def top_features(modelo, feature_names):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[-10:][::-1]

    features = [str(feature_names[i]) for i in indices]
    valores = [round(float(importancias[i]), 4) for i in indices]

    return features, valores


def entrenar_clasificacion(dataset):
    X, y, features, clases = cargar_dataset(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    f_names, f_vals = top_features(model, features)

    return {
        "tipo": "clasificacion",
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)) * 100, 2),
        "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)) * 100, 2),
        "matriz": confusion_matrix(y_test, y_pred).tolist(),
        "features": f_names,
        "importancias": f_vals,
        "clases": [str(c) for c in clases],
        "conteo_predicciones": [int(np.sum(y_pred == i)) for i in range(len(clases))],
        "total_entrenamiento": len(X_train),
        "total_prueba": len(X_test),
        "primeras_probabilidades": np.round(y_proba[:5], 3).tolist(),
    }


def entrenar_regresion(dataset):
    X, y, features, _ = cargar_dataset(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    f_names, f_vals = top_features(model, features)

    return {
        "tipo": "regresion",
        "mae": round(float(mae), 3),
        "mse": round(float(mse), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2) * 100, 2),
        "features": f_names,
        "importancias": f_vals,
        "total_entrenamiento": len(X_train),
        "total_prueba": len(X_test),
        "reales": np.round(y_test[:20], 2).tolist(),
        "predichos": np.round(y_pred[:20], 2).tolist(),
    }


@app.route("/")
def index():
    return render_template("index.html", modelos=MODELOS)


@app.route("/modelo/<modelo_id>")
def modelo(modelo_id):
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
            "resultados": resultados,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/evaluar_credito", methods=["POST"])
def evaluar_credito():
    try:
        data = request.get_json()

        limit_bal = float(data.get("limit_bal", 0))
        age = float(data.get("age", 0))
        pay_0 = float(data.get("pay_0", 0))
        pay_2 = float(data.get("pay_2", 0))
        bill_amt1 = float(data.get("bill_amt1", 0))
        pay_amt1 = float(data.get("pay_amt1", 0))

        if limit_bal <= 0:
            return jsonify({"error": "El monto de crédito debe ser mayor a 0"})
        if age < 18 or age > 80:
            return jsonify({"error": "La edad debe estar entre 18 y 80 años"})

        X, y = make_classification(
            n_samples=2000,
            n_features=6,
            n_informative=5,
            n_redundant=0,
            n_classes=2,
            random_state=42,
        )

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=42,
        )

        model.fit(X, y)

        entrada = np.array([[
            limit_bal / 100000,
            age / 100,
            pay_0,
            pay_2,
            bill_amt1 / 100000,
            pay_amt1 / 10000,
        ]])

        pred = int(model.predict(entrada)[0])
        prob = float(model.predict_proba(entrada)[0][1]) * 100

        return jsonify({
            "riesgo": "Riesgo alto" if pred == 1 else "Riesgo bajo",
            "probabilidad": round(prob, 2),
            "recomendacion": "Revisar historial de pagos antes de aprobar." if pred == 1 else "Cliente con perfil favorable.",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)