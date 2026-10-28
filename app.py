"""
app.py
------
App web (Flask) que usa el MLP para predecir si un estudiante aprueba.

Rutas:
    GET  /          -> página con el formulario
    POST /predecir  -> recibe el formulario y muestra el resultado
    POST /predict   -> ENDPOINT JSON (para Postman, curl, otras apps)
    POST /entrenar  -> vuelve a entrenar el modelo (servicio de entrenamiento)
    GET  /metricas  -> muestra las métricas del modelo (JSON)
    GET  /health    -> sirve para comprobar que la app está viva

Ejecutar en tu computador:
    python app.py
y abrir  http://127.0.0.1:5000
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import entrenar_modelo as em

app = Flask(__name__)

modelo = None
escalador = None


def cargar_o_entrenar(forzar=False):
    """Carga el modelo guardado. Si no existe (o falla), lo entrena."""
    global modelo, escalador
    if not forzar and os.path.exists(em.MODELO_PATH) and os.path.exists(em.ESCALADOR_PATH):
        try:
            modelo = joblib.load(em.MODELO_PATH)
            escalador = joblib.load(em.ESCALADOR_PATH)
            return
        except Exception:
            pass  # si el archivo no sirve, se entrena de nuevo
    em.entrenar()
    modelo = joblib.load(em.MODELO_PATH)
    escalador = joblib.load(em.ESCALADOR_PATH)


def leer_metricas():
    if os.path.exists(em.METRICAS_PATH):
        with open(em.METRICAS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def predecir(datos):
    """datos: diccionario con las 6 características. Devuelve (clase, probabilidad)."""
    fila = pd.DataFrame([[float(datos[c]) for c in em.FEATURES]], columns=em.FEATURES)
    fila_esc = escalador.transform(fila)
    clase = int(modelo.predict(fila_esc)[0])
    prob_aprobar = float(modelo.predict_proba(fila_esc)[0][1])
    return clase, prob_aprobar


def validar(datos):
    """Revisa que estén todos los campos y que los rangos tengan sentido."""
    rangos = {
        "Tareas": (0, 100),
        "Examenes": (0, 100),
        "Asistencia": (0, 100),
        "Estudio_Horas": (0, 100),
        "Sueno_Horas": (0, 24),
        "Extracurriculares": (0, 1),
    }
    for campo, (minimo, maximo) in rangos.items():
        if campo not in datos or str(datos[campo]).strip() == "":
            return f"Falta el campo: {campo}"
        try:
            valor = float(datos[campo])
        except ValueError:
            return f"El campo {campo} debe ser un número"
        if not (minimo <= valor <= maximo):
            return f"{campo} debe estar entre {minimo} y {maximo}"
    return None


@app.route("/")
def inicio():
    return render_template("index.html", resultado=None, error=None, valores={}, metricas=leer_metricas())


@app.route("/predecir", methods=["POST"])
def predecir_formulario():
    datos = request.form.to_dict()
    error = validar(datos)
    if error:
        return render_template("index.html", resultado=None, error=error, valores=datos, metricas=leer_metricas())
    clase, prob = predecir(datos)
    resultado = {
        "texto": "APROBADO" if clase == 1 else "REPROBADO",
        "clase": clase,
        "probabilidad": round(prob * 100, 1),
    }
    return render_template("index.html", resultado=resultado, error=None, valores=datos, metricas=leer_metricas())


@app.route("/predict", methods=["POST"])
def predict_json():
    datos = request.get_json(silent=True)
    if not datos:
        return jsonify({"error": "Envía un JSON con las 6 características"}), 400
    error = validar(datos)
    if error:
        return jsonify({"error": error}), 400
    clase, prob = predecir(datos)
    return jsonify(
        {
            "prediccion": clase,
            "resultado": "Aprobado" if clase == 1 else "Reprobado",
            "probabilidad_aprobar": round(prob, 4),
        }
    )


@app.route("/entrenar", methods=["POST"])
def entrenar_servicio():
    cargar_o_entrenar(forzar=True)
    return jsonify({"mensaje": "Modelo entrenado de nuevo", "metricas": leer_metricas()})


@app.route("/metricas")
def metricas():
    m = leer_metricas()
    return jsonify(m if m else {"mensaje": "Aún no hay métricas"})


@app.route("/health")
def health():
    return jsonify({"estado": "ok"})


# Al arrancar (con python app.py o con gunicorn) se prepara el modelo.
cargar_o_entrenar()

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto, debug=False)
