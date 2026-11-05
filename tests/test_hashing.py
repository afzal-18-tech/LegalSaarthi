"""Tests for src.utils.hashing module."""
import tempfile
from pathlib import Path

import pytest

from src.utils.hashing import sha256_file, sha256_bytes


class TestSha256File:
    """Tests for file hashing."""

    def test_hash_known_content(self, tmp_path):
        """SHA-256 of known content should be deterministic."""
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")

        h = sha256_file(f)
        # Known SHA-256 of "hello world"
        assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_hash_empty_file(self, tmp_path):
        """Should handle empty files."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")

        h = sha256_file(f)
        # Known SHA-256 of empty string
        assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_nonexistent_file(self):
        """Should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            sha256_file("/nonexistent/file.txt")


class TestSha256Bytes:
    """Tests for byte hashing."""

    def test_hash_known_bytes(self):
        h = sha256_bytes(b"hello world")
        assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_hash_empty_bytes(self):
        h = sha256_bytes(b"")
        assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
