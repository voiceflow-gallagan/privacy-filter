from fastapi import APIRouter, Response

from app import model as model_module


router = APIRouter()


@router.get("/health")
def health():
    if model_module.is_loaded():
        state = model_module.get_state()
        return {"status": "ok", "model_loaded": True, "device": state.device}
    return {"status": "ok", "model_loaded": False, "device": None}


@router.get("/ready")
def ready(response: Response):
    if model_module.is_loaded():
        return {"status": "ready"}
    response.status_code = 503
    return {"status": "loading"}
