# Unidad 01 · Ejercicio 02

## Ensambles, reducción dimensional y Green AI

**Asignatura:** INF-8239 Ciencia de Datos II  
**Código:** U01.E02  
**Autor:** Edwin Ramón José Nolasco  
**Repositorio:** https://github.com/rhaynel19/Ejercicio-01-Dataset-p-blico-y-SVM-reproducible

Esta actividad reutiliza el mismo dataset, target y partición aprobados en el Ejercicio 01: Wisconsin Diagnostic Breast Cancer (WDBC), cargado con `sklearn.datasets.load_breast_cancer`.

## Ejecución reproducible

Desde esta carpeta:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH="."
python run_experiment.py
python -m pytest -q
```

Desde la raíz del repositorio:

```powershell
$env:PYTHONPATH="actividad 2"
python "actividad 2/run_experiment.py"
python -m pytest -q "actividad 2/tests"
```

Abra y ejecute `notebooks/01_ensambles_green_ai.ipynb` con el intérprete del entorno virtual.

## Configuraciones comparadas

Se comparan seis configuraciones homogéneas, evaluadas con tres semillas temporales: `42`, `43` y `44`. Para cada repetición se mide F1 macro, accuracy, tiempo de entrenamiento, latencia de inferencia, tamaño serializado y parámetros aproximados.

- SVM.
- SVM con PCA.
- Random Forest.
- Random Forest con PCA.
- HistGradientBoosting.
- HistGradientBoosting con PCA.

La tabla principal usa la mediana de las tres repeticiones.

## Evidencias

- `reports/results_repetitions.csv`: resultados de cada repetición.
- `reports/results_median.csv`: tabla consolidada por mediana.
- `reports/summary.json`: configuración y mejor modelo.
- `reports/pca_2d.png`: visualización PCA.
- `reports/tsne_perplexity_5.png`: primera visualización t-SNE.
- `reports/tsne_perplexity_30.png`: segunda visualización t-SNE.
- `reports/pareto_frontier.png`: frontera coste-desempeño.
- `reports/model_*.joblib`: modelos serializados.
- `reports/entrega_actividad_2.pdf`: informe breve con resultados, figuras y conclusión.
- `model_card.md`: decisiones, limitaciones y uso responsable de IA.

## Green AI y decisión

La selección no se basa solo en F1 macro. También considera tiempo de entrenamiento, latencia de inferencia, tamaño serializado y complejidad. La frontera de Pareto permite identificar configuraciones no dominadas: modelos que no pueden mejorar desempeño sin pagar un costo adicional relevante.
