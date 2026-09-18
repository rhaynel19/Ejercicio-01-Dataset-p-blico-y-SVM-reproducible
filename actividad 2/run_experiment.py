from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.experiment import run_experiment

REPOSITORY = "https://github.com/rhaynel19/Ejercicio-01-Dataset-p-blico-y-SVM-reproducible"


def text_page(pdf_writer, title, sections):
    text_figure = plt.figure(figsize=(8.5, 11))
    text_figure.text(0.08, 0.94, title, fontsize=19, weight="bold", color="#17324d")
    y_position = 0.87
    for heading, body in sections:
        text_figure.text(0.08, y_position, heading, fontsize=12, weight="bold", color="#b24c32")
        y_position -= 0.035
        text_figure.text(0.08, y_position, body, fontsize=10.5, va="top", linespacing=1.5)
        y_position -= 0.055 + body.count("\n") * 0.018
    pdf_writer.savefig(text_figure, bbox_inches="tight")
    plt.close(text_figure)


def image_page(pdf_writer, report_path, title, filename):
    image_figure, image_axis = plt.subplots(figsize=(8.5, 7))
    image_axis.imshow(plt.imread(report_path / filename))
    image_axis.axis("off")
    image_figure.suptitle(title, fontsize=17, weight="bold", color="#17324d")
    image_figure.tight_layout()
    pdf_writer.savefig(image_figure, bbox_inches="tight")
    plt.close(image_figure)


if __name__ == "__main__":
    root = Path(__file__).parent
    report_dir = root / "reports"
    results, median = run_experiment(report_dir)
    best = median.loc[median["f1_macro"].idxmax()]

    with PdfPages(report_dir / "entrega_actividad_2.pdf") as pdf:
        text_page(pdf, "INF-8239 · Unidad 01 · Ejercicio 02", [
            ("Ensambles, reducción dimensional y Green AI", "Autor: Edwin Ramón José Nolasco\nCódigo: U01.E02"),
            ("Repositorio", REPOSITORY),
            ("Objetivo", "Comparar seis configuraciones de SVM, Random Forest y boosting, con y sin PCA, usando tres repeticiones temporales y la mediana como resumen robusto."),
            ("Dataset", "Wisconsin Diagnostic Breast Cancer (WDBC), 569 observaciones y 30 variables numéricas. Se reutiliza el dataset y el target aprobados en el Ejercicio 01."),
            ("Decisión resumida", f"La mejor configuración por F1 macro mediano fue {best['configuration']}, con F1 {best['f1_macro']:.4f}, tiempo de entrenamiento {best['train_seconds']:.3f} s y tamaño serializado {best['model_size_kb']:.1f} KB."),
        ])
        text_page(pdf, "1. Metodología reproducible", [
            ("Partición", "En cada repetición se usa una partición 80/20 estratificada. Las semillas temporales son 42, 43 y 44. El conjunto de prueba permanece separado durante el ajuste."),
            ("Configuraciones", "SVM, SVM + PCA, Random Forest, Random Forest + PCA, HistGradientBoosting y HistGradientBoosting + PCA. El escalamiento y PCA se integran en Pipeline cuando corresponde."),
            ("Métricas", "Se miden F1 macro y accuracy. Para Green AI se registran tiempo de entrenamiento, milisegundos por muestra en inferencia, tamaño serializado en KB y parámetros aproximados."),
            ("Reducción dimensional", "PCA conserva el 95% de la varianza dentro del pipeline. t-SNE se utiliza solo como exploración visual con perplexity 5 y 30; no se utiliza para entrenar ni evaluar modelos."),
            ("Interpretación t-SNE", "Perplexity 5 enfatiza vecindarios locales y puede fragmentar grupos pequeños; perplexity 30 busca una estructura más global y suaviza la visualización. Las dos figuras se comparan como evidencia exploratoria, no como prueba de separación clínica."),
        ])
        figure, axis = plt.subplots(figsize=(8.5, 7))
        axis.axis("off")
        axis.set_title("2. Tabla principal: medianas de tres repeticiones", fontsize=16, weight="bold", color="#17324d")
        columns = ["Configuración", "F1 macro", "Accuracy", "Train (s)", "Inferencia (ms)", "Tamaño (KB)"]
        table_values = [[row["configuration"], f"{row['f1_macro']:.4f}", f"{row['accuracy']:.4f}", f"{row['train_seconds']:.3f}", f"{row['inference_ms_per_sample']:.3f}", f"{row['model_size_kb']:.1f}"] for _, row in median.iterrows()]
        table = axis.table(cellText=[columns] + table_values, loc="upper center", cellLoc="center", colWidths=[0.25, 0.13, 0.13, 0.13, 0.17, 0.14])
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 2)
        axis.text(0.05, 0.38, "Interpretación", fontsize=12, weight="bold", color="#b24c32")
        axis.text(0.05, 0.32, "La mediana reduce la sensibilidad a una sola partición temporal. La elección final debe considerar simultáneamente desempeño y costo, no únicamente el máximo F1.", fontsize=10.5, va="top", wrap=True)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
        image_page(pdf, report_dir, "3. PCA en dos dimensiones", "pca_2d.png")
        image_page(pdf, report_dir, "4. t-SNE con perplexity 5", "tsne_perplexity_5.png")
        image_page(pdf, report_dir, "5. t-SNE con perplexity 30", "tsne_perplexity_30.png")
        image_page(pdf, report_dir, "6. Frontera Pareto: desempeño frente a costo", "pareto_frontier.png")
        text_page(pdf, "7. Conclusión y trazabilidad", [
            ("Conclusión", f"La comparación muestra el compromiso entre discriminación y recursos. Se seleccionó {best['configuration']} como mejor alternativa cuantitativa por F1 macro mediano, pero la decisión debe conservar la frontera Pareto y priorizar configuraciones no dominadas cuando la diferencia de F1 sea pequeña."),
            ("Evidencias", "Los resultados por repetición están en results_repetitions.csv; las medianas en results_median.csv; los modelos están serializados en reports/model_*.joblib."),
            ("IA utilizada", "Se utilizó GitHub Copilot para proponer estructura, explicar errores, sugerir pruebas y revisar redacción. Se verificaron manualmente dataset, particiones, configuraciones, repeticiones, métricas, figuras, archivos serializados y pruebas."),
            ("Reproducción", "Desde actividad 2: python -m pip install -r requirements.txt; $env:PYTHONPATH='.'; python run_experiment.py; python -m pytest -q."),
        ])

    print(f"Mejor configuración: {best['configuration']}")
    print(f"F1 macro mediano: {best['f1_macro']:.4f}")
    print("Reportes y modelos escritos en reports/")
