"""Image processing service for background removal."""

import io
import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize

from app.config import get_settings
from app.ml_models.manager import model_manager
from app.services.statistics import statistics_service

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Service for processing images with background removal models."""

    def __init__(self):
        """Initialize the image processor."""
        self.settings = get_settings()

    def preprocess_image(
        self, image: np.ndarray, model_input_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Preprocess an image for model inference.

        Args:
            image: Input image as numpy array (H, W, C).
            model_input_size: Target size for the model (height, width).

        Returns:
            Preprocessed image tensor.
        """
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]

        # Handle RGBA images
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Convert to tensor and resize
        im_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(
            torch.unsqueeze(im_tensor, 0),
            size=model_input_size,
            mode="bilinear",
            align_corners=False,
        )

        # Normalize
        image_tensor = torch.divide(im_tensor, 255.0)
        image_tensor = normalize(image_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        return image_tensor

    def postprocess_mask(
        self, result: torch.Tensor, original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Postprocess the model output to get the mask.

        Args:
            result: Model output tensor.
            original_size: Original image size (height, width).

        Returns:
            Mask as numpy array (H, W) with values 0-255.
        """
        result = torch.squeeze(
            F.interpolate(result, size=original_size, mode="bilinear", align_corners=False),
            0,
        )

        # Normalize to 0-1
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi + 1e-8)

        # Convert to numpy array
        im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)

        return im_array

    def remove_background(
        self,
        image_bytes: bytes,
        model_name: Optional[str] = None,
        return_mask: bool = False,
    ) -> Tuple[bytes, dict]:
        """Remove background from an image.

        Args:
            image_bytes: Input image as bytes.
            model_name: Model to use. If None, uses the current model.
            return_mask: If True, returns the mask instead of the processed image.

        Returns:
            Tuple of (processed image bytes, metadata dict).

        Raises:
            ValueError: If no model is loaded.
            RuntimeError: If processing fails.
        """
        start_time = time.time()
        target_model = model_name or model_manager.current_model_name

        if target_model is None:
            raise ValueError("No model is loaded. Load a model first.")

        # Ensure model is loaded
        if not model_manager.is_model_loaded(target_model):
            model_manager.load_model(target_model)

        try:
            # Load image
            original_image = Image.open(io.BytesIO(image_bytes))
            original_size = original_image.size  # (width, height)

            # Validate image size
            if max(original_size) > self.settings.max_image_size:
                raise ValueError(
                    f"Image too large. Maximum dimension is {self.settings.max_image_size}px."
                )

            # Convert to RGB if necessary
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Convert to numpy array
            orig_im = np.array(original_image)
            orig_im_size = orig_im.shape[0:2]  # (height, width)

            # Get model and preprocess
            model = model_manager.get_model(target_model)
            model_input_size = (self.settings.model_input_size, self.settings.model_input_size)
            image_tensor = self.preprocess_image(orig_im, model_input_size)
            image_tensor = image_tensor.to(model_manager.device)

            # Run inference
            with torch.no_grad():
                result = model(image_tensor)

            # Postprocess
            mask_array = self.postprocess_mask(result[0][0], orig_im_size)
            mask_image = Image.fromarray(mask_array)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Record statistics
            statistics_service.record_processing(
                model_name=target_model,
                processing_time_ms=processing_time_ms,
                success=True,
                image_size=original_size,
            )

            metadata = {
                "model": target_model,
                "processing_time_ms": round(processing_time_ms, 2),
                "original_size": original_size,
                "device": model_manager.device,
            }

            if return_mask:
                # Return just the mask
                output_buffer = io.BytesIO()
                mask_image.save(output_buffer, format="PNG")
                return output_buffer.getvalue(), metadata

            # Apply mask to original image
            result_image = original_image.copy()
            result_image.putalpha(mask_image)

            # Save to bytes
            output_buffer = io.BytesIO()
            result_image.save(output_buffer, format="PNG")

            return output_buffer.getvalue(), metadata

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            # Record failed processing
            statistics_service.record_processing(
                model_name=target_model or "unknown",
                processing_time_ms=processing_time_ms,
                success=False,
                image_size=(0, 0),
                error_message=error_msg,
            )

            logger.error(f"Background removal failed: {error_msg}")
            raise RuntimeError(f"Background removal failed: {error_msg}")

    def remove_background_pipeline(
        self,
        image_bytes: bytes,
        model_name: Optional[str] = None,
        return_mask: bool = False,
    ) -> Tuple[bytes, dict]:
        """Remove background using the transformers pipeline (alternative method).

        This method is simpler but may be slightly slower than the direct method.

        Args:
            image_bytes: Input image as bytes.
            model_name: Model to use. If None, uses the current model.
            return_mask: If True, returns the mask instead of the processed image.

        Returns:
            Tuple of (processed image bytes, metadata dict).
        """
        start_time = time.time()
        target_model = model_name or model_manager.current_model_name

        if target_model is None:
            raise ValueError("No model is loaded. Load a model first.")

        try:
            # Get pipeline
            pipe = model_manager.get_pipeline(target_model)

            # Load image
            original_image = Image.open(io.BytesIO(image_bytes))
            original_size = original_image.size

            # Validate size
            if max(original_size) > self.settings.max_image_size:
                raise ValueError(
                    f"Image too large. Maximum dimension is {self.settings.max_image_size}px."
                )

            # Process with pipeline
            if return_mask:
                result = pipe(original_image, return_mask=True)
            else:
                result = pipe(original_image)

            processing_time_ms = (time.time() - start_time) * 1000

            # Record statistics
            statistics_service.record_processing(
                model_name=target_model,
                processing_time_ms=processing_time_ms,
                success=True,
                image_size=original_size,
            )

            metadata = {
                "model": target_model,
                "processing_time_ms": round(processing_time_ms, 2),
                "original_size": original_size,
                "device": model_manager.device,
                "method": "pipeline",
            }

            # Convert result to bytes
            output_buffer = io.BytesIO()
            result.save(output_buffer, format="PNG")

            return output_buffer.getvalue(), metadata

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            statistics_service.record_processing(
                model_name=target_model or "unknown",
                processing_time_ms=processing_time_ms,
                success=False,
                image_size=(0, 0),
                error_message=error_msg,
            )

            logger.error(f"Background removal (pipeline) failed: {error_msg}")
            raise RuntimeError(f"Background removal failed: {error_msg}")


# Global processor instance
image_processor = ImageProcessor()

