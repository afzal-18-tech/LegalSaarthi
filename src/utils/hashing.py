"""
LegalSaarthi SHA-256 hashing utility.

Computes file checksums for provenance and integrity validation.
"""
import hashlib
from pathlib import Path


def sha256_file(filepath: str | Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        filepath: Path to file.
        chunk_size: Read buffer size in bytes.

    Returns:
        Lowercase hex digest string.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of byte data.

    Args:
        data: Bytes to hash.

    Returns:
        Lowercase hex digest string.
    """
    return hashlib.sha256(data).hexdigest()
