"""Statistics API endpoints."""

from fastapi import APIRouter, Depends

from app.auth import verify_api_key
from app.services.statistics import statistics_service

router = APIRouter()


@router.get(
    "/stats",
    summary="Get processing statistics",
    description="Get statistics about image processing operations.",
)
async def get_statistics(
    _api_key: str = Depends(verify_api_key),
) -> dict:
    """Get current processing statistics.

    Args:
        _api_key: Validated API key.

    Returns:
        Dictionary with processing statistics.
    """
    return statistics_service.get_statistics()


@router.post(
    "/stats/reset",
    summary="Reset statistics",
    description="Reset all processing statistics to zero.",
)
async def reset_statistics(
    _api_key: str = Depends(verify_api_key),
) -> dict:
    """Reset all statistics.

    Args:
        _api_key: Validated API key.

    Returns:
        Confirmation message.
    """
    statistics_service.reset_statistics()
    return {
        "success": True,
        "message": "Statistics have been reset",
    }

