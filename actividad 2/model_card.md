# Model Card: TinyCNN para dígitos manuscritos

## Identificación

- **Autor:** Edwin Ramón José Nolasco
- **Asignatura:** INF-8239 Ciencia de Datos II
- **Unidad:** 02 · Procesamiento de Lenguaje Natural
- **Código:** U02.E04
- **Modelo:** TinyCNN, una capa convolucional 3×3 con ReLU, aplanado de características y salida softmax.

## Uso previsto

Clasificación demostrativa de imágenes pequeñas de dígitos manuscritos (0–9). El modelo sirve para documentar un flujo reproducible de clasificación visual; no está destinado a decisiones sensibles ni a imágenes fuera de la distribución `load_digits`.

## Datos y partición

Se usa `sklearn.datasets.load_digits`, con 1,797 imágenes en escala de grises de 8×8 píxeles. La intensidad se normaliza a `[0, 1]`. La partición es 80/20, estratificada, con `random_state=42`; el conjunto de prueba se reserva hasta la evaluación final.

## Resultados obtenidos

El baseline mayoritario obtuvo accuracy `0.1000` y F1 macro `0.0182`. La TinyCNN obtuvo accuracy `0.9139` y F1 macro `0.9127`, con 2,970 parámetros y aproximadamente 5 segundos de entrenamiento en CPU. Los valores exactos se conservan en `reports/metrics.json`; también se incluyen matriz de confusión, curvas de pérdida/accuracy y conteo de errores por clase.

## Limitaciones y riesgos

El dataset es pequeño, está preprocesado y sus imágenes son mucho más limpias que entradas reales. La evaluación no garantiza desempeño con escritura, resolución, iluminación o dispositivos diferentes. Los errores entre clases visualmente parecidas deben inspeccionarse antes de concluir que el modelo generaliza.

## Decisión técnica

La CNN se conserva si mejora claramente al baseline y mantiene un costo de entrenamiento aceptable. Si el aumento de complejidad no aporta mejora, el baseline debe preferirse por simplicidad. El notebook registra esta decisión con métricas reproducibles.

## IA utilizada

Se utilizó asistencia de IA para proponer la estructura del experimento, revisar errores y sugerir pruebas. El estudiante verificó el código ejecutándolo en el entorno virtual, revisó las métricas y corrigió cualquier discrepancia antes de presentar resultados. No se usaron referencias ni resultados inventados.
