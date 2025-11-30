"""Model management API endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth import verify_api_key
from app.config import get_settings
from app.ml_models.manager import model_manager

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    loaded: bool
    is_current: bool


class ModelsResponse(BaseModel):
    """Response for list models endpoint."""

    available_models: List[ModelInfo]
    current_model: Optional[str]
    device: str
    cuda_available: bool


class LoadModelRequest(BaseModel):
    """Request to load a model."""

    model_name: str


class LoadModelResponse(BaseModel):
    """Response after loading a model."""

    success: bool
    message: str
    model_name: str
    device: str


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available models",
    description="Get a list of all available models and their status.",
)
async def list_models(
    _api_key: str = Depends(verify_api_key),
) -> ModelsResponse:
    """List all available models and their current status.

    Args:
        _api_key: Validated API key.

    Returns:
        List of models with their load status.
    """
    settings = get_settings()
    model_info = model_manager.get_model_info()

    # If no models configured, show loaded models
    model_names = settings.available_models_list or list(model_info["loaded_models"])
    
    available = [
        ModelInfo(
            name=model_name,
            loaded=model_name in model_info["loaded_models"],
            is_current=model_name == model_info["current_model"],
        )
        for model_name in model_names
    ]

    return ModelsResponse(
        available_models=available,
        current_model=model_info["current_model"],
        device=model_info["device"],
        cuda_available=model_info["cuda_available"],
    )


@router.post(
    "/models/load",
    response_model=LoadModelResponse,
    summary="Load a model",
    description="Load a specific model into memory.",
)
async def load_model(
    request: LoadModelRequest,
    _api_key: str = Depends(verify_api_key),
) -> LoadModelResponse:
    """Load a model into memory.

    Args:
        request: The load model request with model name.
        _api_key: Validated API key.

    Returns:
        Load operation result.
    """
    settings = get_settings()

    # If available_models is configured, enforce whitelist
    if settings.available_models_list and request.model_name not in settings.available_models_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{request.model_name}' is not available. "
            f"Available models: {settings.available_models_list}",
        )

    try:
        model_manager.load_model(request.model_name)
        return LoadModelResponse(
            success=True,
            message=f"Model '{request.model_name}' loaded successfully",
            model_name=request.model_name,
            device=model_manager.device,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@router.post(
    "/models/{model_name}/load",
    response_model=LoadModelResponse,
    summary="Load a specific model",
    description="Load a specific model by name into memory.",
)
async def load_model_by_name(
    model_name: str,
    _api_key: str = Depends(verify_api_key),
) -> LoadModelResponse:
    """Load a specific model by name.

    Args:
        model_name: The name of the model to load.
        _api_key: Validated API key.

    Returns:
        Load operation result.
    """
    settings = get_settings()

    # Handle URL-encoded model names
    from urllib.parse import unquote
    model_name = unquote(model_name)

    # If available_models is configured, enforce whitelist
    if settings.available_models_list and model_name not in settings.available_models_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' is not available. "
            f"Available models: {settings.available_models_list}",
        )

    try:
        model_manager.load_model(model_name)
        return LoadModelResponse(
            success=True,
            message=f"Model '{model_name}' loaded successfully",
            model_name=model_name,
            device=model_manager.device,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@router.post(
    "/models/{model_name}/unload",
    summary="Unload a model",
    description="Unload a model from memory to free resources.",
)
async def unload_model(
    model_name: str,
    _api_key: str = Depends(verify_api_key),
) -> dict:
    """Unload a model from memory.

    Args:
        model_name: The name of the model to unload.
        _api_key: Validated API key.

    Returns:
        Unload operation result.
    """
    from urllib.parse import unquote
    model_name = unquote(model_name)

    if not model_manager.is_model_loaded(model_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' is not loaded.",
        )

    model_manager.unload_model(model_name)

    return {
        "success": True,
        "message": f"Model '{model_name}' unloaded successfully",
    }


@router.post(
    "/models/{model_name}/set-current",
    summary="Set current model",
    description="Set a loaded model as the current default model.",
)
async def set_current_model(
    model_name: str,
    _api_key: str = Depends(verify_api_key),
) -> dict:
    """Set the current active model.

    Args:
        model_name: The name of the model to set as current.
        _api_key: Validated API key.

    Returns:
        Operation result.
    """
    from urllib.parse import unquote
    model_name = unquote(model_name)

    if not model_manager.is_model_loaded(model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' is not loaded. Load it first.",
        )

    model_manager.set_current_model(model_name)

    return {
        "success": True,
        "message": f"Current model set to '{model_name}'",
        "current_model": model_name,
    }

