from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.cnn_digits import run_experiment


REPOSITORY = "https://github.com/rhaynel19/Ejercicio-01-Dataset-p-blico-y-SVM-reproducible"


def add_text_page(pdf_writer, title, sections):
    text_figure = plt.figure(figsize=(8.5, 11))
    text_figure.text(0.08, 0.94, title, fontsize=20, weight="bold", color="#17324d")
    y_position = 0.87
    for heading, body in sections:
        text_figure.text(0.08, y_position, heading, fontsize=12, weight="bold", color="#b24c32")
        y_position -= 0.035
        text_figure.text(0.08, y_position, body, fontsize=10.5, va="top", linespacing=1.5)
        y_position -= 0.055 + 0.018 * body.count("\n")
    pdf_writer.savefig(text_figure, bbox_inches="tight")
    plt.close(text_figure)


def add_image_page(pdf_writer, report_path, title, filename):
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
    metrics = run_experiment(report_dir)

    with PdfPages(report_dir / "entrega_actividad_2.pdf") as pdf:
        add_text_page(pdf, "INF-8239 · Actividad 2", [
            ("Clasificador visual con CNN y Model Card", "Entrega individual\nAutor: Edwin Ramón José Nolasco\nCódigo: U02.E04\nUnidad 02 · Procesamiento de Lenguaje Natural"),
            ("Repositorio", REPOSITORY),
            ("Síntesis ejecutiva", "Se compara un baseline mayoritario con una TinyCNN entrenada en CPU sobre dígitos manuscritos. La CNN se conserva porque mejora ampliamente al baseline y el entrenamiento tarda aproximadamente cinco segundos."),
            ("Resultado principal", f"Baseline: accuracy {metrics['baseline']['accuracy']:.4f}, F1 macro {metrics['baseline']['f1_macro']:.4f}\nTinyCNN: accuracy {metrics['cnn']['accuracy']:.4f}, F1 macro {metrics['cnn']['f1_macro']:.4f}"),
        ])

        add_text_page(pdf, "1. Auditoría, partición y decisiones", [
            ("Datos", "Dataset load_digits de scikit-learn: 1,797 imágenes en escala de grises, 8×8 píxeles y 10 clases. Las intensidades se normalizan de 0–16 a [0, 1]. No se requieren descargas externas."),
            ("Auditoría", "La entrada es numérica, tiene forma consistente y no contiene valores ausentes. El objetivo está separado de las imágenes. La distribución se conserva mediante partición estratificada."),
            ("Partición reproducible", f"Entrenamiento: {metrics['split']['train']} observaciones. Prueba: {metrics['split']['test']} observaciones. test_size=0.20, random_state={metrics['split']['random_state']}, stratify=target. El conjunto de prueba se reserva para la evaluación final."),
            ("Arquitectura y decisión", "TinyCNN: convolución válida 3×3 con filtros aprendibles, ReLU, aplanado de características y softmax. Se prefirió esta arquitectura ligera para mantener el costo bajo y conservar información espacial. Se rechazaría la CNN si no superara al baseline."),
        ])

        figure, axis = plt.subplots(figsize=(8.5, 6.5))
        axis.axis("off")
        axis.set_title("2. Resultados y costo", fontsize=17, weight="bold", color="#17324d", pad=18)
        table_data = [
            ["Modelo", "Accuracy", "F1 macro", "Tamaño / costo"],
            ["Baseline mayoritario", f"{metrics['baseline']['accuracy']:.4f}", f"{metrics['baseline']['f1_macro']:.4f}", "sin parámetros entrenables"],
            ["TinyCNN", f"{metrics['cnn']['accuracy']:.4f}", f"{metrics['cnn']['f1_macro']:.4f}", f"{metrics['cnn']['parameters']} parámetros; {metrics['cnn']['train_seconds']:.2f} s"],
        ]
        table = axis.table(cellText=table_data, loc="upper center", cellLoc="center", colWidths=[0.28, 0.16, 0.16, 0.34])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        axis.text(0.05, 0.48, "Interpretación", fontsize=12, weight="bold", color="#b24c32")
        axis.text(0.05, 0.42, "La CNN supera al baseline en ambas métricas. El F1 macro confirma que la mejora no depende solo de la clase más frecuente. El costo es pequeño para una ejecución CPU y queda registrado en metrics.json.", fontsize=11, va="top", linespacing=1.6)
        axis.text(0.05, 0.23, "Evidencias", fontsize=12, weight="bold", color="#b24c32")
        axis.text(0.05, 0.17, "training_curves.png · confusion_matrix.png · errors_by_class.png · metrics.json", fontsize=11, va="top")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        add_image_page(pdf, report_dir, "3. Curvas de entrenamiento", "training_curves.png")
        add_image_page(pdf, report_dir, "4. Matriz de confusión", "confusion_matrix.png")
        add_image_page(pdf, report_dir, "4.2 Conteo de errores por clase", "errors_by_class.png")

        add_text_page(pdf, "5. Model Card, IA y reproducibilidad", [
            ("Uso previsto y limitaciones", "El modelo es demostrativo para clasificar dígitos pequeños de load_digits. No debe usarse como sistema general de reconocimiento manuscrito ni para decisiones sensibles. El dataset es pequeño y preprocesado; la generalización a otras cámaras, resoluciones o poblaciones no está garantizada."),
            ("Uso responsable de IA", "Herramienta utilizada: asistencia de GitHub Copilot en VS Code. Prompts relevantes: solicitar la estructura reproducible de la actividad, pedir pruebas para la CNN, revisar errores de ejecución y verificar el cumplimiento de la rúbrica. Se verificaron manualmente el código, las métricas, las figuras, la notebook y la suite de pruebas."),
            ("Correcciones realizadas", "Se corrigió el broadcast del sesgo convolucional, se reemplazó el promedio global por aplanado para conservar información espacial, se aumentó el entrenamiento a 100 épocas tras comprobar el desempeño, se convirtió la notebook al formato JSON solicitado y se amplió este PDF con evidencia visual."),
            ("Reproducción", "Desde actividad 2: pip install -r requirements.txt; $env:PYTHONPATH='.'; python run_experiment.py; python -m pytest -q. Resultado verificado: 3 pruebas aprobadas, accuracy CNN 0.9139 y F1 macro 0.9127."),
        ])

    print(f"CNN accuracy: {metrics['cnn']['accuracy']:.4f}")
    print(f"CNN F1 macro: {metrics['cnn']['f1_macro']:.4f}")
    print("Reportes escritos en reports/")
