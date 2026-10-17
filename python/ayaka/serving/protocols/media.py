"""Bounded base64 images. Network image sources are never fetched."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

from .error import ApiError

MAX_HTTP_IMAGE_BYTES = 8 * 1024 * 1024
MAX_HTTP_BODY_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ImageInput:
    media_type: str
    data: bytes


def decode_image_data_url(value: str, maximum: int = MAX_HTTP_IMAGE_BYTES) -> ImageInput:
    if not isinstance(value, str) or not value.startswith("data:"):
        raise ApiError("image source must be a base64 data URL; network URLs are not supported")
    metadata, separator, encoded = value[5:].partition(",")
    if not separator:
        raise ApiError("image data URL must use base64 encoding")
    fields = metadata.split(";")
    if not fields[0].startswith("image/") or len(fields[0]) <= 6:
        raise ApiError("image data URL must declare an image media type")
    if fields[1:] != ["base64"]:
        raise ApiError("image data URL must use base64 encoding")
    if ((len(encoded) + 3) // 4) * 3 > maximum + 2:
        raise ApiError(f"image exceeds the {maximum} byte decoded-size limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ApiError("image data URL contains invalid base64") from exc
    if not data:
        raise ApiError("image data URL decoded to an empty payload")
    if len(data) > maximum:
        raise ApiError(f"image exceeds the {maximum} byte decoded-size limit")
    return ImageInput(fields[0], data)
