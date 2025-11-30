"""Model manager for loading and switching Hugging Face models."""

import logging
from typing import Dict, Optional, Any
import threading

import torch
from transformers import AutoModelForImageSegmentation, pipeline

from app.config import get_settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading, switching, and caching of background removal models."""

    def __init__(self):
        """Initialize the model manager."""
        self._loaded_models: Dict[str, Any] = {}
        self._pipelines: Dict[str, Any] = {}
        self._current_model: Optional[str] = None
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model manager initialized. Device: {self._device}")

    @property
    def device(self) -> str:
        """Get the current device (cuda or cpu)."""
        return self._device

    @property
    def current_model_name(self) -> Optional[str]:
        """Get the name of the currently active model."""
        return self._current_model

    @property
    def loaded_models(self) -> Dict[str, Any]:
        """Get dictionary of loaded models."""
        return self._loaded_models.copy()

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded.

        Args:
            model_name: The Hugging Face model identifier.

        Returns:
            True if the model is loaded, False otherwise.
        """
        return model_name in self._loaded_models

    def load_model(self, model_name: str) -> bool:
        """Load a model into memory.

        Args:
            model_name: The Hugging Face model identifier.

        Returns:
            True if the model was loaded successfully.

        Raises:
            ValueError: If the model name is not in the available models list.
            RuntimeError: If model loading fails.
        """
        settings = get_settings()

        # If available_models is configured, enforce the whitelist
        if settings.available_models_list and model_name not in settings.available_models_list:
            raise ValueError(
                f"Model '{model_name}' is not in the available models list. "
                f"Available: {settings.available_models_list}"
            )

        with self._lock:
            if model_name in self._loaded_models:
                logger.info(f"Model '{model_name}' is already loaded")
                self._current_model = model_name
                return True

            try:
                logger.info(f"Loading model: {model_name}")

                # Load model using transformers
                model = AutoModelForImageSegmentation.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=settings.model_cache_dir,
                )
                model.to(self._device)
                model.eval()

                # Create pipeline for easy inference
                pipe = pipeline(
                    "image-segmentation",
                    model=model_name,
                    trust_remote_code=True,
                    device=0 if self._device == "cuda" else -1,
                    model_kwargs={"cache_dir": settings.model_cache_dir},
                )

                self._loaded_models[model_name] = model
                self._pipelines[model_name] = pipe
                self._current_model = model_name

                logger.info(f"Model '{model_name}' loaded successfully on {self._device}")
                return True

            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {e}")
                raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    def get_model(self, model_name: Optional[str] = None) -> Any:
        """Get a loaded model.

        Args:
            model_name: The model to retrieve. If None, returns the current model.

        Returns:
            The loaded model instance.

        Raises:
            ValueError: If no model is loaded or the specified model isn't loaded.
        """
        target = model_name or self._current_model

        if target is None:
            raise ValueError("No model is currently loaded")

        if target not in self._loaded_models:
            raise ValueError(f"Model '{target}' is not loaded")

        return self._loaded_models[target]

    def get_pipeline(self, model_name: Optional[str] = None) -> Any:
        """Get the pipeline for a loaded model.

        Args:
            model_name: The model pipeline to retrieve. If None, returns current.

        Returns:
            The pipeline instance for the model.

        Raises:
            ValueError: If no model is loaded or the specified model isn't loaded.
        """
        target = model_name or self._current_model

        if target is None:
            raise ValueError("No model is currently loaded")

        if target not in self._pipelines:
            raise ValueError(f"Pipeline for model '{target}' is not available")

        return self._pipelines[target]

    def set_current_model(self, model_name: str) -> bool:
        """Set the current active model.

        Args:
            model_name: The model to set as active.

        Returns:
            True if successful.

        Raises:
            ValueError: If the model isn't loaded.
        """
        if model_name not in self._loaded_models:
            raise ValueError(
                f"Model '{model_name}' is not loaded. Load it first with load_model()."
            )

        self._current_model = model_name
        logger.info(f"Current model set to: {model_name}")
        return True

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory.

        Args:
            model_name: The model to unload.

        Returns:
            True if the model was unloaded.
        """
        with self._lock:
            if model_name not in self._loaded_models:
                logger.warning(f"Model '{model_name}' is not loaded")
                return False

            # Remove model and pipeline
            del self._loaded_models[model_name]
            if model_name in self._pipelines:
                del self._pipelines[model_name]

            # Clear current model if it was unloaded
            if self._current_model == model_name:
                self._current_model = None

            # Force garbage collection for GPU memory
            if self._device == "cuda":
                torch.cuda.empty_cache()

            logger.info(f"Model '{model_name}' unloaded")
            return True

    def unload_all(self) -> None:
        """Unload all models from memory."""
        with self._lock:
            model_names = list(self._loaded_models.keys())
            for model_name in model_names:
                del self._loaded_models[model_name]
                if model_name in self._pipelines:
                    del self._pipelines[model_name]

            self._current_model = None

            if self._device == "cuda":
                torch.cuda.empty_cache()

            logger.info("All models unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models and current state.

        Returns:
            Dictionary with model manager state information.
        """
        settings = get_settings()
        return {
            "device": self._device,
            "cuda_available": torch.cuda.is_available(),
            "current_model": self._current_model,
            "loaded_models": list(self._loaded_models.keys()),
            "available_models": settings.available_models_list,
        }


# Global model manager instance
model_manager = ModelManager()

