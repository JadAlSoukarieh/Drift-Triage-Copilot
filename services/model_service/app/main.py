from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from services.model_service.app.routes.health import router as health_router
from services.model_service.app.routes.predict import router as predict_router

app = FastAPI(
    title="Drift Triage Co-Pilot Model Service",
    version="0.1.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "details": exc.errors(),
        },
    )


app.include_router(health_router)
app.include_router(predict_router)

