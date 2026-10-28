# Taller MLP – ¿Aprobará el estudiante?

## Archivos
- `MLP_TALLER.ipynb`: notebook con todo el proceso, resultados y conclusiones.
- `entrenar_modelo.py`: servicio que entrena el MLP y guarda modelo, escalador y métricas.
- `app.py`: app web Flask (formulario + endpoint JSON).
- `templates/index.html`: página del formulario.
- `dataset_estudiantes_balanceado.csv`: datos.
- `requirements.txt` y `Procfile`: para desplegar.

## Correr en tu computador (Windows, cmd)
```
cd /d D:\Datos\Documentos\Taller_MLP
venv\Scripts\activate
python app.py
```
Abre http://127.0.0.1:5000  (la primera vez entrena el modelo solo).

## Probar el endpoint JSON
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"Tareas\":85,\"Examenes\":80,\"Asistencia\":95,\"Estudio_Horas\":14,\"Sueno_Horas\":7,\"Extracurriculares\":1}"
```

## Desplegar en Render (link público)
1. Sube esta carpeta a un repositorio de GitHub.
2. En render.com: New > Web Service > conecta el repositorio.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
5. Tu endpoint será: `https://TU-APP.onrender.com/predict`
