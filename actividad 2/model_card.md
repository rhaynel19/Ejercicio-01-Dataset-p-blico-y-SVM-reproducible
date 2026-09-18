# Model Card · Ensambles WDBC

## Identificación

- **Asignatura:** INF-8239 Ciencia de Datos II
- **Unidad:** 01 · Modelos avanzados, reducción dimensional y Green AI
- **Código:** U01.E02
- **Autor:** Edwin Ramón José Nolasco
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

## Uso previsto

Comparar modelos clásicos de clasificación binaria sobre variables numéricas de WDBC y documentar el compromiso entre desempeño, costo computacional y tamaño del artefacto. Es una actividad académica y no un sistema clínico.

## Datos y partición

Se usan 569 observaciones y 30 variables numéricas cargadas mediante `load_breast_cancer`. Se utiliza una partición 80/20 estratificada para cada repetición, con semillas 42, 43 y 44. El conjunto de prueba de cada repetición se mantiene separado durante el ajuste.

## Modelos

Se comparan SVM, SVM con PCA, Random Forest, Random Forest con PCA, HistGradientBoosting y HistGradientBoosting con PCA. El escalamiento y la reducción se incorporan en `Pipeline` cuando corresponde.

## Métricas y Green AI

Se reportan F1 macro, accuracy, tiempo de entrenamiento, latencia de inferencia, tamaño serializado y cantidad aproximada de parámetros. La decisión se toma con la mediana de tres repeticiones y la frontera de Pareto de F1 frente a costo.

## Limitaciones

WDBC es un dataset pequeño y clásico; no representa necesariamente poblaciones actuales ni condiciones clínicas reales. Una alta puntuación no equivale a validación clínica. PCA puede reducir dimensionalidad, pero también puede eliminar información útil para algunos modelos. t-SNE sirve para exploración visual y no debe interpretarse como una evaluación del clasificador.

## Uso de IA

Se utilizó GitHub Copilot en VS Code para proponer estructura, explicar errores, sugerir pruebas y revisar la redacción. Se verificaron manualmente los datos, la partición, las seis configuraciones, las tres repeticiones, los CSV, las figuras, los modelos serializados y las pruebas. Se corrigieron errores de la actividad anterior, se reemplazó completamente la CNN por el flujo de ensambles solicitado y se comprobó la ejecución final con el entorno virtual.
