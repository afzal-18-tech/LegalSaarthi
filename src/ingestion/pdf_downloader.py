"""
LegalSaarthi PDF downloader.

Downloads PDFs from resolved URLs with retry, rate limiting,
validation, and structured error logging.
"""
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.ingestion.source_resolver import ResolutionResult
from src.utils.hashing import sha256_file
from src.utils.logging_setup import get_logger

logger = get_logger("ingestion.downloader")


class DownloadResult:
    """Result of a single PDF download attempt."""

    __slots__ = (
        "case_id",
        "filename",
        "url",
        "local_path",
        "sha256",
        "file_size",
        "status",
        "error",
        "method",
        "attempts",
        "accessed_at",
    )

    def __init__(
        self,
        case_id: str = "",
        filename: str = "",
        url: str = "",
        local_path: str = "",
        sha256: str = "",
        file_size: int = 0,
        status: str = "pending",
        error: str = "",
        method: str = "",
        attempts: int = 0,
        accessed_at: str = "",
    ):
        self.case_id = case_id
        self.filename = filename
        self.url = url
        self.local_path = local_path
        self.sha256 = sha256
        self.file_size = file_size
        self.status = status
        self.error = error
        self.method = method
        self.attempts = attempts
        self.accessed_at = accessed_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "filename": self.filename,
            "url": self.url,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "file_size": self.file_size,
            "status": self.status,
            "error": self.error,
            "method": self.method,
            "attempts": self.attempts,
            "accessed_at": self.accessed_at,
        }


def validate_pdf(filepath: Path, min_size: int = 1024) -> tuple[bool, str]:
    """Validate that a file is a valid PDF.

    Checks:
    - File exists
    - File size > min_size
    - Starts with PDF magic bytes (%PDF-)

    Args:
        filepath: Path to the file.
        min_size: Minimum acceptable file size in bytes.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not filepath.exists():
        return False, "File does not exist"

    size = filepath.stat().st_size
    if size < min_size:
        return False, f"File too small: {size} bytes (min: {min_size})"

    with open(filepath, "rb") as f:
        header = f.read(5)

    if header != b"%PDF-":
        return False, f"Invalid PDF signature: {header!r}"

    return True, ""


class PDFDownloader:
    """Downloads PDFs with retry, rate limiting, and validation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        acq = config.get("acquisition", {})
        self.timeout = acq.get("timeout_seconds", 60)
        self.max_retries = acq.get("max_retries", 3)
        self.backoff_factor = acq.get("backoff_factor", 2.0)
        self.rate_limit_rps = acq.get("rate_limit_rps", 2.0)
        self.user_agent = acq.get("user_agent", "LegalSaarthi-Research/0.1")

        val = config.get("validation", {})
        self.min_file_size = val.get("min_file_size", 1024)
        self.check_signature = val.get("check_pdf_signature", True)
        self.compute_sha256 = val.get("compute_sha256", True)

        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        if self.rate_limit_rps <= 0:
            return
        min_interval = 1.0 / self.rate_limit_rps
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def download_one(
        self,
        resolution: ResolutionResult,
        output_dir: Path,
    ) -> DownloadResult:
        """Download a single PDF from a resolved URL.

        Args:
            resolution: ResolutionResult with status='resolved'.
            output_dir: Base directory for downloads.

        Returns:
            DownloadResult with status and metadata.
        """
        result = DownloadResult(
            case_id=resolution.case_id,
            filename=resolution.filename,
            url=resolution.url,
            method=resolution.method,
        )

        if resolution.status != "resolved":
            result.status = "skipped"
            result.error = f"Resolution status: {resolution.status} - {resolution.error}"
            return result

        # Create year subdirectory
        year_dir = output_dir / f"year={resolution.year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        local_path = year_dir / resolution.filename

        # Skip if already downloaded and valid
        if local_path.exists():
            is_valid, err = validate_pdf(local_path, self.min_file_size)
            if is_valid:
                result.status = "already_exists"
                result.local_path = str(local_path)
                result.file_size = local_path.stat().st_size
                if self.compute_sha256:
                    result.sha256 = sha256_file(local_path)
                result.accessed_at = datetime.now(timezone.utc).isoformat()
                logger.debug(f"Already exists: {resolution.filename}")
                return result
            else:
                logger.warning(
                    f"Existing file invalid ({err}), re-downloading: {resolution.filename}"
                )
                local_path.unlink()

        # Download with retry
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            result.attempts = attempt
            try:
                self._rate_limit()
                logger.debug(
                    f"Downloading {resolution.filename} (attempt {attempt})"
                )

                resp = self._session.get(
                    resolution.url,
                    timeout=self.timeout,
                    stream=True,
                )
                resp.raise_for_status()

                # Stream to file
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Validate
                is_valid, err = validate_pdf(local_path, self.min_file_size)
                if not is_valid:
                    last_error = f"Validation failed: {err}"
                    logger.warning(
                        f"Download validation failed for {resolution.filename}: {err}"
                    )
                    if local_path.exists():
                        local_path.unlink()
                    continue

                # Success
                result.status = "success"
                result.local_path = str(local_path)
                result.file_size = local_path.stat().st_size
                result.accessed_at = datetime.now(timezone.utc).isoformat()

                if self.compute_sha256:
                    result.sha256 = sha256_file(local_path)

                logger.info(
                    f"Downloaded: {resolution.filename} "
                    f"({result.file_size} bytes, sha256={result.sha256[:12]}...)"
                )
                return result

            except requests.HTTPError as e:
                last_error = f"HTTP {e.response.status_code}: {e}"
                logger.warning(
                    f"HTTP error downloading {resolution.filename} "
                    f"(attempt {attempt}): {last_error}"
                )
            except requests.ConnectionError as e:
                last_error = f"Connection error: {e}"
                logger.warning(
                    f"Connection error for {resolution.filename} "
                    f"(attempt {attempt}): {last_error}"
                )
            except requests.Timeout:
                last_error = "Request timeout"
                logger.warning(
                    f"Timeout downloading {resolution.filename} "
                    f"(attempt {attempt})"
                )
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.error(
                    f"Unexpected error downloading {resolution.filename}: {e}"
                )
                break

            # Exponential backoff before retry
            if attempt < self.max_retries:
                wait = self.backoff_factor ** attempt
                logger.debug(f"Retrying in {wait:.1f}s...")
                time.sleep(wait)

        # All retries exhausted
        result.status = "failed"
        result.error = last_error
        result.accessed_at = datetime.now(timezone.utc).isoformat()
        return result

    def download_batch(
        self,
        resolutions: list[ResolutionResult],
        output_dir: Path,
        limit: int | None = None,
    ) -> list[DownloadResult]:
        """Download a batch of resolved PDFs.

        Args:
            resolutions: List of ResolutionResult objects.
            output_dir: Base directory for downloads.
            limit: Optional max number to download.

        Returns:
            List of DownloadResult objects.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        # Filter to resolved only
        resolved = [r for r in resolutions if r.status == "resolved"]
        if limit:
            resolved = resolved[:limit]

        logger.info(f"Starting batch download: {len(resolved)} files")

        for i, resolution in enumerate(resolved):
            result = self.download_one(resolution, output_dir)
            results.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == len(resolved):
                success = sum(
                    1 for r in results
                    if r.status in ("success", "already_exists")
                )
                failed = sum(1 for r in results if r.status == "failed")
                logger.info(
                    f"Progress: {i + 1}/{len(resolved)} "
                    f"(success={success}, failed={failed})"
                )

        success = sum(
            1 for r in results if r.status in ("success", "already_exists")
        )
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")

        logger.info(
            f"Batch complete: {success} success, {failed} failed, "
            f"{skipped} skipped out of {len(results)} total"
        )
        return results
