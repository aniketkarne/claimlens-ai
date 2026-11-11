"""Document loader - handles PDF and image uploads."""

import os
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pdf2image
from fastapi import HTTPException, UploadFile

from app.config import get_settings


class DocumentLoader:
    """Handles loading and conversion of uploaded documents."""

    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = self.settings.upload_path
        self.max_upload_mb = self.settings.max_upload_mb
        self.retention = timedelta(hours=self.settings.upload_retention_hours)

    async def save_upload(self, file: UploadFile, claim_id: str) -> Path:
        """
        Save an uploaded file to the upload directory.

        Args:
            file: The uploaded FastAPI file
            claim_id: Unique claim identifier for naming

        Returns:
            Path to the saved file

        Raises:
            HTTPException: 413 when the upload exceeds the configured limit
        """
        # Opportunistic retention sweep — keeps the upload directory bounded
        # even when earlier runs crashed before cleanup could happen.
        try:
            self.sweep_stale()
        except Exception:
            pass  # best effort

        # Determine file extension
        content_type = file.content_type
        if content_type == "application/pdf":
            extension = ".pdf"
        elif content_type in ("image/jpeg", "image/jpg"):
            extension = ".jpg"
        elif content_type == "image/png":
            extension = ".png"
        else:
            extension = ".bin"

        # Create filename
        filename = f"{claim_id}{extension}"
        file_path = self.upload_dir / filename

        # Ensure directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Write file in chunks — never hold the whole upload in memory, and
        # refuse oversized bodies instead of letting them exhaust RAM.
        max_bytes = self.max_upload_mb * 1024 * 1024
        received = 0
        try:
            with open(file_path, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload too large: limit is {self.max_upload_mb} MB",
                        )
                    f.write(chunk)
        except Exception:
            # Never leave a partial claim document behind
            try:
                file_path.unlink()
            except OSError:
                pass
            raise

        return file_path

    def sweep_stale(self, max_age: Optional[timedelta] = None) -> List[str]:
        """
        Delete claim artifacts older than the retention window.

        Uploaded claim forms carry full PII (names, addresses, plates,
        policy numbers) and the derived artifacts (PDF page renders,
        preprocessed images) do too. Nothing else removes them, so anything
        left behind by a crashed or interrupted run is swept here.

        Args:
            max_age: Retention window; defaults to the configured
                upload_retention_hours.

        Returns:
            Paths that were removed.
        """
        if max_age is None:
            max_age = self.retention
        cutoff = time.time() - max_age.total_seconds()
        removed: List[str] = []
        if not self.upload_dir.exists():
            return removed
        for path in self.upload_dir.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed.append(str(path))
            except OSError:
                continue  # best effort — file may vanish mid-sweep
        return removed

    def convert_pdf_to_images(
        self,
        pdf_path: Path,
        dpi: int = 200,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None
    ) -> List[str]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for conversion (default 200)
            first_page: First page to convert (1-indexed, None = first)
            last_page: Last page to convert (None = all or first only for POC)

        Returns:
            List of paths to generated images
        """
        image_paths = []

        try:
            # For POC, only process first page
            # Set first_page and last_page to 1 to only convert first page
            if first_page is None:
                first_page = 1
            if last_page is None:
                last_page = 1

            # Convert PDF to images
            images = pdf2image.convert_from_path(
                str(pdf_path),
                dpi=dpi,
                first_page=first_page,
                last_page=last_page
            )

            # Save images
            base_name = pdf_path.stem
            for i, image in enumerate(images):
                image_filename = f"{base_name}_page_{first_page + i}.png"
                image_path = self.upload_dir / image_filename
                image.save(str(image_path), "PNG")
                image_paths.append(str(image_path))

        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")

        return image_paths

    def load_image(self, image_path: str) -> str:
        """
        Validate that a file is a valid image.

        Args:
            image_path: Path to the image file

        Returns:
            Path to the validated image
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Check extension
        valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid image format: {path.suffix}")

        return str(path)

    def cleanup(self, file_path: Path) -> None:
        """
        Remove a temporary file.

        Args:
            file_path: Path to the file to remove
        """
        try:
            if file_path.exists():
                os.remove(file_path)
        except Exception:
            pass  # Best effort cleanup
