"""
entrenar_modelo.py
------------------
SERVICIO DE ENTRENAMIENTO del MLP.

Hace lo mismo que el notebook, pero en un script que la app web puede usar:
1. Carga el CSV
2. Imputa vacíos con la MEDIANA de cada columna
3. Quita Calificacion_Final (evita fuga de datos)
4. Divide en entrenamiento (80%) y prueba (20%)
5. Estandariza (media 0, varianza 1)
6. Entrena el MLP
7. Evalúa y guarda el modelo + el escalador + las métricas

Uso desde la terminal:
    python entrenar_modelo.py
"""
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset_estudiantes_balanceado.csv")
MODELO_PATH = os.path.join(BASE_DIR, "modelo_mlp.pkl")
ESCALADOR_PATH = os.path.join(BASE_DIR, "escalador.pkl")
METRICAS_PATH = os.path.join(BASE_DIR, "metricas.json")

# Columnas que el modelo recibe como ENTRADA (X).
# OJO: Calificacion_Final NO está, porque de ella sale "Aprobado" (fuga de datos).
FEATURES = [
    "Tareas",
    "Examenes",
    "Asistencia",
    "Estudio_Horas",
    "Sueno_Horas",
    "Extracurriculares",
]
TARGET = "Aprobado"


def entrenar():
    # 1) Cargar
    df = pd.read_csv(CSV_PATH)

    # 2) Imputar con la mediana de cada columna
    df = df.fillna(df.median(numeric_only=True))

    # 3) Separar entradas (X) y respuesta (y)
    X = df[FEATURES]
    y = df[TARGET]

    # 4) Dividir 80% entrenamiento / 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) Estandarizar: el escalador "aprende" solo con entrenamiento
    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    # 6) Crear y entrenar el MLP (2 capas ocultas: 16 y 8 neuronas)
    modelo = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
    )
    modelo.fit(X_train_esc, y_train)

    # 7) Evaluar con el set de prueba
    y_pred = modelo.predict(X_test_esc)
    metricas = {
        "accuracy_prueba": round(float(accuracy_score(y_test, y_pred)), 4),
        "accuracy_entrenamiento": round(
            float(accuracy_score(y_train, modelo.predict(X_train_esc))), 4
        ),
        "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
        "reporte": classification_report(y_test, y_pred, output_dict=True),
        "n_entrenamiento": int(len(X_train)),
        "n_prueba": int(len(X_test)),
    }

    # 8) Guardar todo
    joblib.dump(modelo, MODELO_PATH)
    joblib.dump(escalador, ESCALADOR_PATH)
    with open(METRICAS_PATH, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    return metricas


if __name__ == "__main__":
    m = entrenar()
    print("Modelo entrenado y guardado.")
    print("Accuracy entrenamiento:", m["accuracy_entrenamiento"])
    print("Accuracy prueba       :", m["accuracy_prueba"])
    print("Matriz de confusión   :", m["matriz_confusion"])
