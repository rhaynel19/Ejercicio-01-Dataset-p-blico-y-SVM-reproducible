# Entrega individual INF-8239

Repositorio reproducible de LAB01 y LAB02: auditoría de datos, comparación de datasets y experimento de clasificación con SVM. El repositorio local es la evidencia versionada; no hay remoto configurado todavía. Al publicar el proyecto, sustituir este texto por la URL pública.

## Pregunta y aprobación

**Pregunta aprobada:** ¿un SVM con variables estandarizadas puede distinguir diagnósticos malignos y benignos en el Wisconsin Diagnostic Breast Cancer (WDBC) con mejor F1 macro que un baseline mayoritario?

| Dataset | Procedencia y carga | Dimensión / tarea | Licencia o uso | Decisión |
|---|---|---:|---|---|
| Wisconsin Diagnostic Breast Cancer (WDBC) | UCI Machine Learning Repository; reproducible mediante `sklearn.datasets.load_breast_cancer` | 569 x 30; clasificación binaria | UCI, ficha CC BY 4.0 | **Aprobado como dataset principal**: variables numéricas, pregunta binaria clara y tamaño manejable |
| Iris | UCI Machine Learning Repository; reproducible mediante `sklearn.datasets.load_iris` | 150 x 4; clasificación multiclase | UCI, ficha CC BY 4.0 | **Aprobado como alternativa comparativa**: útil para validar el flujo, pero menos exigente para la pregunta clínica |

El análisis ejecuta WDBC. La carga desde scikit-learn evita depender de una descarga manual: la versión de las dependencias queda registrada en el entorno y el dataset se reconstruye con una llamada determinista. La fuente original es UCI; las etiquetas de scikit-learn son `0=malignant` y `1=benign`.

## Reproducibilidad

En PowerShell, desde la raíz:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m pytest -q
```

Abrir `notebooks/01_svm_guiada.ipynb` con el kernel `.venv` y ejecutar todas las celdas de principio a fin. El notebook carga el dataset incluido en scikit-learn, ejecuta la auditoría, separa entrenamiento/prueba con `stratify=y` y `random_state=42`, ajusta el baseline y el SVM, optimiza hiperparámetros con validación cruzada estratificada y guarda las evidencias en `reports/`.

## Auditoría y prevención de fuga

- `X` tiene 569 filas y 30 variables numéricas; `y` tiene 212 casos de clase 0 y 357 de clase 1.
- No se encontraron duplicados ni valores ausentes; el target no forma parte de `X`.
- El conjunto de prueba contiene 114 filas y se reserva antes del ajuste.
- `StandardScaler` está dentro de un `Pipeline` junto con `SVC`; por ello cada transformación se aprende solo con los datos de entrenamiento de cada partición.
- `GridSearchCV` usa `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` y se ajusta exclusivamente sobre `X_train, y_train`. El conjunto de prueba se usa una sola vez para la evaluación final.

## Resultados principales

| Modelo | F1 macro | ROC-AUC | Observación |
|---|---:|---:|---|
| Dummy `most_frequent` | 0.3871 | -- | Baseline de referencia |
| SVM base | 0.9812 | 0.9950 | `C=1`, `gamma=scale` |
| SVM optimizado | **0.9812** | **0.9977** | `C=10`, `gamma=0.01` |

Los resultados completos están en [reports/svm_cv_results.csv](reports/svm_cv_results.csv) y [reports/svm_summary.json](reports/svm_summary.json). Las figuras son [reports/svm_confusion_matrix.png](reports/svm_confusion_matrix.png) y [reports/svm_roc_curve.png](reports/svm_roc_curve.png). El modelo serializado está en `reports/svm_best.joblib`; permanece ignorado por Git para evitar versionar binarios generados.

## Código y pruebas

- [src/inf8239_u01/models.py](src/inf8239_u01/models.py) expone `build_svm`, un pipeline reutilizable y leakage-safe.
- [tests/test_environment.py](tests/test_environment.py) verifica el entorno.
- [tests/test_models.py](tests/test_models.py) verifica predicciones, validación de `C` y composición del pipeline.

Resultado actual de la suite: `4 passed`.

## Conclusión

El experimento responde afirmativamente a la pregunta planteada: un SVM con estandarización integrada supera ampliamente al baseline mayoritario y mantiene un desempeño muy alto sobre el conjunto de prueba. El baseline alcanza un F1 macro de 0.3871 porque siempre predice la clase dominante; por tanto, no ofrece una discriminación útil entre diagnósticos. El SVM base alcanza F1 macro de 0.9812 y ROC-AUC de 0.9950. La búsqueda de hiperparámetros selecciona `C=10` y `gamma=0.01`, con ROC-AUC de 0.9977 y el mismo F1 macro redondeado a cuatro decimales. La mejora más visible aparece en la capacidad de ordenar correctamente los casos, aunque el F1 no cambia con el redondeo.

La conclusión es defendible porque la evaluación separa el conjunto de prueba antes del ajuste y reserva ese conjunto para una sola medición final. Además, el escalamiento ocurre dentro del pipeline, de modo que no se calculan medias ni desviaciones usando observaciones de prueba. La validación cruzada estratificada conserva la proporción de clases y permite comparar configuraciones sin convertir el conjunto final en parte del proceso de selección. La auditoría no encontró ausentes, duplicados ni una columna objetivo filtrada dentro de las variables predictoras.

Aun así, el resultado no debe interpretarse como una validación clínica. WDBC es un dataset pequeño y clásico, sus observaciones no representan necesariamente una población actual y la partición aleatoria no modela variaciones entre centros, instrumentos o pacientes. El siguiente paso sería validar el pipeline con una fuente externa y reportar intervalos de incertidumbre, sensibilidad específica y análisis de errores por clase. En esta entrega, la principal evidencia es metodológica: el flujo es reproducible, las decisiones están registradas y el código probado separa transformación, entrenamiento y evaluación de manera clara.

## Entrega PDF

El PDF breve es [reports/entrega_inf8239.pdf](reports/entrega_inf8239.pdf) e incluye la ficha comparativa, la tabla de resultados, las figuras y esta conclusión.
