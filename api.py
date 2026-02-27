"""
FastAPI REST API para el clasificador local de discurso de odio.

Uso:
    uvicorn api:app --reload                        # desarrollo
    uvicorn api:app --host 0.0.0.0 --port 8000      # producción

Endpoints:
    POST /classify   — clasifica un tweet
    GET  /health     — health check
    GET  /docs       — Swagger UI (auto-generado)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from classifier import DIRECTORIO_MODELO_DEFAULT, LABELS, classify, _cargar_modelo

# Executor dedicado para la inferencia (evita bloquear el event loop)
_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warming: carga el modelo antes de recibir requests
    print(f"Cargando modelo desde '{DIRECTORIO_MODELO_DEFAULT}'...")
    await asyncio.get_event_loop().run_in_executor(
        _executor,
        lambda: _cargar_modelo(DIRECTORIO_MODELO_DEFAULT),
    )
    print("Modelo cargado. API lista.")
    yield
    _executor.shutdown(wait=False)


app = FastAPI(
    title="Tweet Wing — Clasificador de Discurso de Odio",
    description=(
        "Clasifica tweets en español según 10 categorías de discurso de odio "
        "usando BETO fine-tuned localmente (sin API externa)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    tweet: str
    threshold: float = 0.5

    model_config = {"json_schema_extra": {
        "example": {
            "tweet": "Estos inmigrantes le sacan el trabajo a los argentinos",
            "threshold": 0.5,
        }
    }}


class ClassifyResponse(BaseModel):
    labels: dict[str, bool]
    probabilities: dict[str, float]
    detected_labels: list[str]
    is_hateful: bool


@app.post("/classify", response_model=ClassifyResponse)
async def classify_tweet(request: ClassifyRequest) -> ClassifyResponse:
    if not request.tweet.strip():
        raise HTTPException(status_code=400, detail="El campo 'tweet' no puede estar vacío")
    if not (0.0 < request.threshold < 1.0):
        raise HTTPException(
            status_code=400,
            detail="El umbral debe estar entre 0 y 1 (exclusivo)",
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: classify(tweet=request.tweet, threshold=request.threshold),
    )

    labels_dict = {label: getattr(result, label) for label in LABELS}
    detected = [label for label in LABELS if labels_dict[label]]

    return ClassifyResponse(
        labels=labels_dict,
        probabilities={k: round(v, 4) for k, v in result._probabilities.items()},
        detected_labels=detected,
        is_hateful=result.HATEFUL,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
