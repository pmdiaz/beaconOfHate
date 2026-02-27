"""
Script de evaluación del clasificador local contra el dataset
piuba-bigdata/contextualized_hate_speech de HuggingFace.

Uso:
    python evaluate.py                          # 50 ejemplos del split test
    python evaluate.py --n 200 --split validation
    python evaluate.py --n 500 --threshold 0.4
    python evaluate.py --model-dir ruta/al/modelo
"""

import argparse

from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score

from classifier import DIRECTORIO_MODELO_DEFAULT, LABELS, classify


def evaluate(
    n_samples: int = 50,
    split: str = "test",
    threshold: float = 0.5,
    model_dir: str = DIRECTORIO_MODELO_DEFAULT,
) -> None:
    print(f"Cargando dataset (split={split})...")
    dataset = load_dataset("piuba-bigdata/contextualized_hate_speech", split=split)

    n = min(n_samples, len(dataset))
    samples = dataset.select(range(n))
    print(f"Evaluando {n} ejemplos (umbral={threshold})...\n")

    y_true = {label: [] for label in LABELS}
    y_pred = {label: [] for label in LABELS}
    errors = 0

    for i, example in enumerate(samples):
        tweet = example["text"]

        if (i + 1) % 100 == 0 or i == 0:
            print(f"[{i + 1}/{n}] {tweet[:70]}...")

        try:
            result = classify(tweet=tweet, threshold=threshold, model_dir=model_dir)
            for label in LABELS:
                y_true[label].append(int(bool(example[label])))
                y_pred[label].append(int(getattr(result, label)))
        except Exception as e:
            print(f"  Error en ejemplo {i + 1}: {e}")
            errors += 1
            continue

    processed = n - errors
    print(f"\n{'=' * 60}")
    print(f"RESULTADOS — {processed}/{n} tweets clasificados")
    print(f"{'=' * 60}\n")

    macro_f1s = []
    for label in LABELS:
        if not y_true[label]:
            continue
        n_positivos = sum(y_true[label])
        print(f"--- {label} ({n_positivos}/{len(y_true[label])} positivos en ground truth) ---")
        print(classification_report(
            y_true[label],
            y_pred[label],
            target_names=["No", "Sí"],
            zero_division=0,
        ))
        macro_f1s.append(f1_score(y_true[label], y_pred[label], average="macro", zero_division=0))

    if macro_f1s:
        print(f"Macro F1 promedio (todas las categorías): {sum(macro_f1s) / len(macro_f1s):.3f}")
    if errors:
        print(f"\nErrores durante la evaluación: {errors}/{n}")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el clasificador local sobre el dataset")
    parser.add_argument(
        "--n", type=int, default=50,
        help="Número de ejemplos a evaluar (default: 50)",
    )
    parser.add_argument(
        "--split", default="test",
        choices=["train", "test", "dev"],
        help="Split del dataset a usar (default: test)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Umbral de decisión para las predicciones (default: 0.5)",
    )
    parser.add_argument(
        "--model-dir", default=DIRECTORIO_MODELO_DEFAULT,
        help=f"Directorio del modelo fine-tuned (default: {DIRECTORIO_MODELO_DEFAULT})",
    )
    args = parser.parse_args()
    evaluate(args.n, args.split, args.threshold, args.model_dir)


if __name__ == "__main__":
    main()
