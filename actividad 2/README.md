# Actividad 2 · Clasificador visual con CNN y Model Card

Entrega ejecutable para INF-8239 Ciencia de Datos II, Unidad 02. Autor: Edwin Ramón José Nolasco. Código: U02.E04.

Repositorio: https://github.com/rhaynel19/Ejercicio-01-Dataset-p-blico-y-SVM-reproducible

## Ejecución en `.venv`

Desde esta carpeta:

```powershell
..\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH="."
python run_experiment.py
python -m pytest -q
```

Si la carpeta se ejecuta desde la raíz del repositorio, use:

```powershell
$env:PYTHONPATH="actividad 2"
python "actividad 2/run_experiment.py"
python -m pytest -q "actividad 2/tests"
```

Abra `notebooks/01_cnn_model_card.ipynb`, seleccione el intérprete del `.venv` y ejecute todas las celdas. El script y el notebook escriben las evidencias en `reports/`.

## Evidencias

- `reports/metrics.json`: métricas de baseline y CNN, costo y tamaño.
- `reports/training_curves.png`: pérdida y accuracy durante el entrenamiento.
- `reports/confusion_matrix.png`: errores y aciertos por clase.
- `reports/errors_by_class.png`: conteo de errores por dígito.
- `reports/entrega_actividad_2.pdf`: síntesis visual breve.
- `model_card.md`: uso previsto, datos, limitaciones y decisión técnica.

La solución usa `load_digits` de scikit-learn para evitar descargas manuales. La CNN NumPy tiene una capa convolucional 3×3 con filtros aprendibles, ReLU, aplanado de características y softmax; por eso el experimento sigue siendo ligero y reproducible en CPU.
