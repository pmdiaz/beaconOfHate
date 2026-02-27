"""
Clasificador local de discurso de odio usando BETO fine-tuned.
Puede usarse como librería o como CLI.

Uso:
    python classifier.py "El tweet a clasificar"
    python classifier.py "El tweet" --json
    python classifier.py "El tweet" --threshold 0.4 --probs
    python classifier.py "El tweet" --model-dir ruta/al/modelo
"""

import argparse
import json
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = [
    "HATEFUL", "CALLS", "WOMEN", "LGBTI", "RACISM",
    "CLASS", "POLITICS", "DISABLED", "APPEARANCE", "CRIMINAL",
]

DIRECTORIO_MODELO_DEFAULT = "modelo_hate_speech"

# Singletons para carga lazy (evita recargar en cada llamada)
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSequenceClassification] = None
_model_dir_cargado: Optional[str] = None


def _cargar_modelo(model_dir: str):
    """Carga tokenizer y modelo una sola vez (lazy, con singleton)."""
    global _tokenizer, _model, _model_dir_cargado
    if _model is None or _model_dir_cargado != model_dir:
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        _model.eval()
        _model_dir_cargado = model_dir


# ---------------------------------------------------------------------------
# Clase de resultado
# ---------------------------------------------------------------------------

class HateSpeechClassification:
    """
    Resultado de clasificación multi-label.
    Preserva la interfaz getattr(result, label) y model_dump_json().
    """

    def __init__(self, labels: dict[str, bool], probabilities: dict[str, float]):
        for label, value in labels.items():
            setattr(self, label, value)
        self._probabilities = probabilities

    def model_dump_json(self, indent: int = 2) -> str:
        data = {label: getattr(self, label) for label in LABELS}
        data["probabilities"] = {k: round(v, 4) for k, v in self._probabilities.items()}
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def model_dump(self) -> dict:
        data = {label: getattr(self, label) for label in LABELS}
        data["probabilities"] = {k: round(v, 4) for k, v in self._probabilities.items()}
        return data


# ---------------------------------------------------------------------------
# Función principal de clasificación
# ---------------------------------------------------------------------------

def classify(
    tweet: str,
    threshold: float = 0.5,
    model_dir: str = DIRECTORIO_MODELO_DEFAULT,
) -> HateSpeechClassification:
    """
    Clasifica un tweet usando el modelo local fine-tuned.

    Args:
        tweet: Texto del tweet a clasificar.
        threshold: Umbral de decisión para convertir probabilidades a booleanos (default: 0.5).
        model_dir: Directorio donde está guardado el modelo fine-tuned.

    Returns:
        HateSpeechClassification con atributos booleanos por label y probabilidades.
    """
    _cargar_modelo(model_dir)

    inputs = _tokenizer(
        tweet,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )

    with torch.inference_mode():
        logits = _model(**inputs).logits

    probs = torch.sigmoid(logits).squeeze().tolist()
    if isinstance(probs, float):
        probs = [probs]

    labels_bool = {label: probs[i] >= threshold for i, label in enumerate(LABELS)}
    labels_probs = {label: probs[i] for i, label in enumerate(LABELS)}

    return HateSpeechClassification(labels=labels_bool, probabilities=labels_probs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clasificador local de discurso de odio en tweets en español"
    )
    parser.add_argument("tweet", help="Texto del tweet a clasificar")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Umbral de decisión para las predicciones (default: 0.5)",
    )
    parser.add_argument(
        "--model-dir", default=DIRECTORIO_MODELO_DEFAULT,
        help=f"Directorio del modelo fine-tuned (default: {DIRECTORIO_MODELO_DEFAULT})",
    )
    parser.add_argument("--json", action="store_true", help="Output en formato JSON")
    parser.add_argument("--probs", action="store_true", help="Mostrar probabilidades por label")
    args = parser.parse_args()

    result = classify(
        tweet=args.tweet,
        threshold=args.threshold,
        model_dir=args.model_dir,
    )

    if args.json:
        print(result.model_dump_json(indent=2))
        return

    print(f"\nTweet: {args.tweet}")
    print("-" * 60)
    detected = [label for label in LABELS if getattr(result, label)]
    if detected:
        print(f"Categorías detectadas: {', '.join(detected)}")
    else:
        print("Sin discurso de odio detectado")

    if args.probs:
        print("\nProbabilidades:")
        for label in LABELS:
            prob = result._probabilities[label]
            marcador = "✓" if getattr(result, label) else " "
            print(f"  [{marcador}] {label:<12}: {prob:.3f}")


if __name__ == "__main__":
    main()
