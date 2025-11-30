"""Background removal API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from app.auth import verify_api_key
from app.services.processor import image_processor

router = APIRouter()


@router.post(
    "/remove-background",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Processed image with transparent background",
        },
        400: {"description": "Invalid image or request"},
        401: {"description": "Missing or invalid API key"},
        500: {"description": "Processing error"},
    },
    summary="Remove background from image",
    description="Upload an image to remove its background. Returns a PNG with transparency.",
)
async def remove_background(
    file: UploadFile = File(..., description="Image file to process"),
    model: Optional[str] = Form(None, description="Model to use (optional)"),
    return_mask: bool = Form(False, description="Return mask instead of processed image"),
    _api_key: str = Depends(verify_api_key),
) -> Response:
    """Remove background from an uploaded image.

    Args:
        file: The image file to process.
        model: Optional model name to use. If not provided, uses the default model.
        return_mask: If True, returns just the alpha mask.
        _api_key: Validated API key (injected by dependency).

    Returns:
        PNG image with transparent background or mask.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Only images are supported.",
        )

    # Read file content
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Process image
    try:
        result_bytes, metadata = image_processor.remove_background(
            image_bytes=image_bytes,
            model_name=model,
            return_mask=return_mask,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # Return processed image
    return Response(
        content=result_bytes,
        media_type="image/png",
        headers={
            "X-Processing-Time-Ms": str(metadata["processing_time_ms"]),
            "X-Model-Used": metadata["model"],
            "X-Original-Width": str(metadata["original_size"][0]),
            "X-Original-Height": str(metadata["original_size"][1]),
        },
    )


@router.post(
    "/remove-background/base64",
    responses={
        200: {"description": "Processed image as base64"},
        400: {"description": "Invalid image or request"},
        401: {"description": "Missing or invalid API key"},
        500: {"description": "Processing error"},
    },
    summary="Remove background (base64 response)",
    description="Remove background and return result as base64-encoded PNG.",
)
async def remove_background_base64(
    file: UploadFile = File(..., description="Image file to process"),
    model: Optional[str] = Form(None, description="Model to use (optional)"),
    return_mask: bool = Form(False, description="Return mask instead of processed image"),
    _api_key: str = Depends(verify_api_key),
) -> dict:
    """Remove background and return base64-encoded result.

    Args:
        file: The image file to process.
        model: Optional model name to use.
        return_mask: If True, returns just the alpha mask.
        _api_key: Validated API key.

    Returns:
        JSON with base64-encoded image and metadata.
    """
    import base64

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Only images are supported.",
        )

    # Read file content
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Process image
    try:
        result_bytes, metadata = image_processor.remove_background(
            image_bytes=image_bytes,
            model_name=model,
            return_mask=return_mask,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # Encode to base64
    base64_image = base64.b64encode(result_bytes).decode("utf-8")

    return {
        "image": base64_image,
        "format": "png",
        "metadata": {
            "model": metadata["model"],
            "processing_time_ms": metadata["processing_time_ms"],
            "original_size": {
                "width": metadata["original_size"][0],
                "height": metadata["original_size"][1],
            },
            "device": metadata["device"],
        },
    }

