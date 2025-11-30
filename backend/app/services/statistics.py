"""Statistics tracking service for OpenBackground."""

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessingRecord:
    """Record of a single image processing request."""

    timestamp: float
    model_name: str
    processing_time_ms: float
    success: bool
    image_size: tuple
    error_message: Optional[str] = None


@dataclass
class Statistics:
    """Aggregated statistics for the service."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time_ms: float = 0.0
    requests_per_model: Dict[str, int] = field(default_factory=dict)
    recent_records: List[ProcessingRecord] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time in milliseconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_processing_time_ms / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for API response."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_processing_time_ms": round(self.average_processing_time_ms, 2),
            "success_rate_percent": round(self.success_rate, 2),
            "requests_per_model": self.requests_per_model.copy(),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "recent_requests": [
                {
                    "timestamp": datetime.fromtimestamp(r.timestamp).isoformat(),
                    "model": r.model_name,
                    "processing_time_ms": round(r.processing_time_ms, 2),
                    "success": r.success,
                    "image_size": r.image_size,
                }
                for r in self.recent_records[-10:]  # Last 10 requests
            ],
        }


class StatisticsService:
    """Service for tracking and persisting processing statistics."""

    def __init__(self, max_recent_records: int = 100):
        """Initialize the statistics service.

        Args:
            max_recent_records: Maximum number of recent records to keep in memory.
        """
        self._stats = Statistics()
        self._lock = threading.Lock()
        self._max_recent = max_recent_records
        self._db_conn: Optional[sqlite3.Connection] = None

        settings = get_settings()
        if settings.enable_stats_persistence:
            self._init_database(settings.stats_db_path)

    def _init_database(self, db_path: str) -> None:
        """Initialize SQLite database for persistence.

        Args:
            db_path: Path to the SQLite database file.
        """
        try:
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            self._db_conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = self._db_conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    processing_time_ms REAL NOT NULL,
                    success INTEGER NOT NULL,
                    image_width INTEGER,
                    image_height INTEGER,
                    error_message TEXT
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON processing_records(timestamp)
            """)

            self._db_conn.commit()
            logger.info(f"Statistics database initialized at {db_path}")

            # Load existing stats from database
            self._load_stats_from_db()

        except Exception as e:
            logger.error(f"Failed to initialize statistics database: {e}")
            self._db_conn = None

    def _load_stats_from_db(self) -> None:
        """Load aggregate statistics from database."""
        if self._db_conn is None:
            return

        try:
            cursor = self._db_conn.cursor()

            # Get aggregate stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = 1 THEN processing_time_ms ELSE 0 END) as total_time
                FROM processing_records
            """)
            row = cursor.fetchone()

            if row:
                self._stats.total_requests = row[0]
                self._stats.successful_requests = row[1] or 0
                self._stats.failed_requests = row[0] - (row[1] or 0)
                self._stats.total_processing_time_ms = row[2] or 0.0

            # Get per-model stats
            cursor.execute("""
                SELECT model_name, COUNT(*) as count
                FROM processing_records
                GROUP BY model_name
            """)
            for row in cursor.fetchall():
                self._stats.requests_per_model[row[0]] = row[1]

            logger.info("Statistics loaded from database")

        except Exception as e:
            logger.error(f"Failed to load statistics from database: {e}")

    def record_processing(
        self,
        model_name: str,
        processing_time_ms: float,
        success: bool,
        image_size: tuple,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a processing request.

        Args:
            model_name: Name of the model used.
            processing_time_ms: Processing time in milliseconds.
            success: Whether the processing was successful.
            image_size: Tuple of (width, height) of the processed image.
            error_message: Error message if processing failed.
        """
        record = ProcessingRecord(
            timestamp=time.time(),
            model_name=model_name,
            processing_time_ms=processing_time_ms,
            success=success,
            image_size=image_size,
            error_message=error_message,
        )

        with self._lock:
            # Update in-memory stats
            self._stats.total_requests += 1
            if success:
                self._stats.successful_requests += 1
                self._stats.total_processing_time_ms += processing_time_ms
            else:
                self._stats.failed_requests += 1

            # Update per-model count
            if model_name not in self._stats.requests_per_model:
                self._stats.requests_per_model[model_name] = 0
            self._stats.requests_per_model[model_name] += 1

            # Add to recent records
            self._stats.recent_records.append(record)
            if len(self._stats.recent_records) > self._max_recent:
                self._stats.recent_records = self._stats.recent_records[-self._max_recent:]

            # Persist to database if enabled
            if self._db_conn is not None:
                self._persist_record(record)

    def _persist_record(self, record: ProcessingRecord) -> None:
        """Persist a record to the database.

        Args:
            record: The processing record to persist.
        """
        try:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO processing_records 
                (timestamp, model_name, processing_time_ms, success, image_width, image_height, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp,
                    record.model_name,
                    record.processing_time_ms,
                    1 if record.success else 0,
                    record.image_size[0],
                    record.image_size[1],
                    record.error_message,
                ),
            )
            self._db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist record: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics.

        Returns:
            Dictionary with current statistics.
        """
        with self._lock:
            return self._stats.to_dict()

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._stats = Statistics()

            if self._db_conn is not None:
                try:
                    cursor = self._db_conn.cursor()
                    cursor.execute("DELETE FROM processing_records")
                    self._db_conn.commit()
                    logger.info("Statistics reset")
                except Exception as e:
                    logger.error(f"Failed to reset database statistics: {e}")


# Global statistics service instance
statistics_service = StatisticsService()

