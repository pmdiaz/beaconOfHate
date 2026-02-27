"""
Fine-tuning de BETO (bert-base-spanish-wwm-cased) para clasificación multi-label
de discurso de odio sobre el dataset piuba-bigdata/contextualized_hate_speech.

Estrategia CPU:
  Fase 1 (default): backbone congelado, solo entrena la cabeza lineal (~7,690 params).
                    ~70 minutos en CPU para el dataset completo.
  Fase 2 (--unfreeze-layers N): descongela las últimas N capas transformer.
                    ~3-4 horas en CPU.

Uso:
    python finetune.py                                  # 3 epochs, dataset completo
    python finetune.py --max-samples 2000 --epochs 1   # smoke test ~5 min
    python finetune.py --unfreeze-layers 2              # fase 2
    python finetune.py --lr 3e-4 --epochs 5
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABELS = [
    "HATEFUL", "CALLS", "WOMEN", "LGBTI", "RACISM",
    "CLASS", "POLITICS", "DISABLED", "APPEARANCE", "CRIMINAL",
]
NUM_LABELS = len(LABELS)
MODELO_BASE = "dccuchile/bert-base-spanish-wwm-cased"
DIRECTORIO_SALIDA = "modelo_hate_speech"


# ---------------------------------------------------------------------------
# Tokenización
# ---------------------------------------------------------------------------

def tokenizar_ejemplo(ejemplo, tokenizer, max_length: int = 128):
    """Tokeniza el campo 'text' y convierte los labels a float32."""
    codificado = tokenizer(
        ejemplo["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # el collator aplica padding dinámico
    )
    # Labels como vector float para BCEWithLogitsLoss
    codificado["labels"] = [float(bool(ejemplo[label])) for label in LABELS]
    return codificado


# ---------------------------------------------------------------------------
# Data collator multi-label
# ---------------------------------------------------------------------------

@dataclass
class MultiLabelDataCollator:
    """Collator que hace padding dinámico y convierte labels a float32."""
    tokenizer: AutoTokenizer
    max_length: int = 128

    def __call__(self, features):
        # Separar labels del resto
        labels = [f.pop("labels") for f in features]

        # Padding dinámico sobre input_ids / attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.stack(labels).to(dtype=torch.float32)
        return batch


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convertir logits a predicciones binarias con umbral 0.5
    probs = 1 / (1 + np.exp(-logits))  # sigmoid manual (numpy)
    preds = (probs >= 0.5).astype(int)
    labels_int = labels.astype(int)

    f1_por_label = {}
    macro_f1s = []
    for i, label in enumerate(LABELS):
        f1 = f1_score(labels_int[:, i], preds[:, i], average="macro", zero_division=0)
        f1_por_label[f"f1_{label.lower()}"] = round(f1, 4)
        macro_f1s.append(f1)

    f1_por_label["f1_macro"] = round(float(np.mean(macro_f1s)), 4)
    return f1_por_label


# ---------------------------------------------------------------------------
# Congelado / descongelado del backbone
# ---------------------------------------------------------------------------

def congelar_backbone(model, unfreeze_layers: int = 0):
    """
    Congela todo el encoder de BETO. Si unfreeze_layers > 0,
    descongela las últimas N capas transformer.
    """
    # Congelar embeddings y encoder completo
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False
    # Pooler siempre entrenado (conecta con la cabeza)
    for param in model.bert.pooler.parameters():
        param.requires_grad = True

    if unfreeze_layers > 0:
        total_capas = len(model.bert.encoder.layer)
        capas_a_descongelar = model.bert.encoder.layer[total_capas - unfreeze_layers:]
        for capa in capas_a_descongelar:
            for param in capa.parameters():
                param.requires_grad = True
        print(f"  Descongeladas las últimas {unfreeze_layers} capas transformer")

    # Cabeza de clasificación siempre entrenada
    for param in model.classifier.parameters():
        param.requires_grad = True

    params_entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_totales = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {params_entrenables:,} / {params_totales:,} "
          f"({100 * params_entrenables / params_totales:.1f}%)")


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------

def entrenar(
    epochs: int = 3,
    max_samples: Optional[int] = None,
    unfreeze_layers: int = 0,
    lr: float = 2e-3,
    output_dir: str = DIRECTORIO_SALIDA,
):
    print(f"Cargando tokenizer: {MODELO_BASE}")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    print("Cargando dataset piuba-bigdata/contextualized_hate_speech...")
    dataset_completo = load_dataset("piuba-bigdata/contextualized_hate_speech")

    # Columnas necesarias (text + 10 labels)
    columnas_a_eliminar = [
        col for col in dataset_completo["train"].column_names
        if col not in ["text"] + LABELS
    ]

    # Tokenizar
    print("Tokenizando...")
    dataset_tokenizado = dataset_completo.map(
        lambda ej: tokenizar_ejemplo(ej, tokenizer),
        remove_columns=columnas_a_eliminar,
        batched=False,
    )
    dataset_tokenizado.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_ds = dataset_tokenizado["train"]
    val_ds = dataset_tokenizado["dev"]

    if max_samples is not None:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        val_val = min(max_samples // 5, len(val_ds))
        val_ds = val_ds.select(range(max(val_val, 100)))
        print(f"  Limitado a {len(train_ds)} train / {len(val_ds)} validation")

    # Modelo
    print(f"Cargando modelo: {MODELO_BASE}")
    id2label = {i: label for i, label in enumerate(LABELS)}
    label2id = {label: i for i, label in enumerate(LABELS)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    print("Congelando backbone...")
    congelar_backbone(model, unfreeze_layers)

    # Data collator
    collator = MultiLabelDataCollator(tokenizer=tokenizer)

    # TrainingArguments optimizados para CPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        use_cpu=True,
        dataloader_num_workers=0,
        bf16=False,
        fp16=False,
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print(f"\nIniciando entrenamiento — {epochs} epoch(s), lr={lr}")
    print(f"Train: {len(train_ds)} ejemplos | Val: {len(val_ds)} ejemplos\n")
    trainer.train()

    print(f"\nGuardando modelo en '{output_dir}'...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modelo guardado. Para clasificar: python classifier.py \"tu tweet\"")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning de BETO para clasificación de discurso de odio"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Número de epochs de entrenamiento (default: 3)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limitar el dataset para pruebas (ej: 2000 para smoke test)",
    )
    parser.add_argument(
        "--unfreeze-layers", type=int, default=0,
        help="Descongelar las últimas N capas transformer del backbone (default: 0)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-3,
        help="Learning rate (default: 2e-3 para backbone congelado, 2e-5 para fase 2)",
    )
    parser.add_argument(
        "--output-dir", default=DIRECTORIO_SALIDA,
        help=f"Directorio de salida del modelo (default: {DIRECTORIO_SALIDA})",
    )
    args = parser.parse_args()

    entrenar(
        epochs=args.epochs,
        max_samples=args.max_samples,
        unfreeze_layers=args.unfreeze_layers,
        lr=args.lr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
