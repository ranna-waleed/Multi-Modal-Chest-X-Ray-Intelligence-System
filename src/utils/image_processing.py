"""
Image Processing Utilities
===========================
Handles CXR image loading, preprocessing, and format conversion.
Supports: JPEG, PNG, and DICOM (.dcm) formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


def load_image(
    path: Union[str, Path],
    target_size: Optional[tuple[int, int]] = None,
    enhance_contrast: bool = False,
) -> Image.Image:
    """
    Load a chest X-ray image from disk.

    Supports JPEG, PNG, and DICOM formats.

    Args:
        path: Path to image file
        target_size: Optional (width, height) to resize to
        enhance_contrast: Apply CLAHE-like contrast enhancement

    Returns:
        RGB PIL Image
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".dcm", ".dicom"):
        image = _load_dicom(path)
    else:
        image = Image.open(path).convert("RGB")

    if target_size:
        image = image.resize(target_size, Image.LANCZOS)

    if enhance_contrast:
        image = _enhance_cxr(image)

    return image


def _load_dicom(path: Path) -> Image.Image:
    """Load DICOM file and convert to RGB PIL Image."""
    try:
        import pydicom
        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Normalize to [0, 255]
        pixel_min, pixel_max = pixel_array.min(), pixel_array.max()
        if pixel_max > pixel_min:
            pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255
        pixel_array = pixel_array.astype(np.uint8)

        # Handle photometric interpretation
        if hasattr(ds, "PhotometricInterpretation"):
            if ds.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = 255 - pixel_array  # Invert

        # Convert to RGB
        if pixel_array.ndim == 2:
            image = Image.fromarray(pixel_array, mode="L").convert("RGB")
        elif pixel_array.ndim == 3:
            image = Image.fromarray(pixel_array)
        else:
            raise ValueError(f"Unexpected pixel array shape: {pixel_array.shape}")

        return image

    except ImportError:
        logger.error("pydicom not installed. Run: pip install pydicom")
        raise
    except Exception as e:
        logger.error(f"Failed to load DICOM {path}: {e}")
        raise


def _enhance_cxr(image: Image.Image) -> Image.Image:
    """Apply mild contrast enhancement suitable for CXR viewing."""
    # Convert to grayscale for enhancement, then back to RGB
    gray = image.convert("L")
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(1.5)
    # Slight sharpening
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))
    return enhanced.convert("RGB")


def preprocess_for_model(
    image: Image.Image,
    model_type: str = "medgemma",
    max_size: int = 1024,
) -> Image.Image:
    """
    Preprocess image for a specific model.

    Args:
        image: Input PIL Image
        model_type: "medgemma" | "clip" | "colpali"
        max_size: Maximum dimension

    Returns:
        Preprocessed PIL Image
    """
    image = image.convert("RGB")

    # Resize if too large (preserve aspect ratio)
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    return image


def create_comparison_grid(
    images: list[Image.Image],
    labels: Optional[list[str]] = None,
    cols: int = 2,
) -> Image.Image:
    """
    Create a grid image from multiple images for comparison display.

    Args:
        images: List of PIL Images
        labels: Optional text labels
        cols: Number of columns

    Returns:
        Combined grid PIL Image
    """
    from PIL import ImageDraw

    if not images:
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    # Standardize sizes
    max_w = max(img.size[0] for img in images)
    max_h = max(img.size[1] for img in images)
    target = (min(max_w, 512), min(max_h, 512))

    resized = [img.resize(target, Image.LANCZOS) for img in images]

    rows = (len(resized) + cols - 1) // cols
    padding = 10
    grid_w = cols * target[0] + (cols + 1) * padding
    grid_h = rows * (target[1] + 20) + (rows + 1) * padding  # +20 for label

    grid = Image.new("RGB", (grid_w, grid_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid)

    for idx, img in enumerate(resized):
        row = idx // cols
        col = idx % cols
        x = col * (target[0] + padding) + padding
        y = row * (target[1] + 20 + padding) + padding

        grid.paste(img, (x, y))

        if labels and idx < len(labels):
            draw.text((x, y + target[1] + 2), labels[idx], fill=(50, 50, 50))

    return grid
