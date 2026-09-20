"""Use case: draw what the model saw on a photo that is already stored."""

from dataclasses import dataclass

from loadvision.application.ports import ExplanationKind, ImageStorage, LoadExplainer
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.imaging.validator import ImageLimits, prepare_image


@dataclass(frozen=True, slots=True, kw_only=True)
class ExplainPhoto:
    """Draws a map of what the model saw on a stored photo.

    The photo is decoded exactly as it was when it was assessed, so the map lies over the very
    pixels the prediction was made on.

    Attributes:
        storage: Where the original photo bytes are kept.
        limits: Size and format limits of photos.
    """

    storage: ImageStorage
    limits: ImageLimits

    def __call__(self, image_sha256: str, kind: ExplanationKind, explainer: LoadExplainer) -> bytes:
        """Returns the map as PNG bytes.

        Args:
            image_sha256: Lowercase hex SHA-256 of the photo.
            kind: Which map to draw.
            explainer: Model to draw with; the caller passes one snapshot of the active model.

        Returns:
            PNG bytes.

        Raises:
            LoadvisionError: `ROUTE_NOT_FOUND` if the photo is not stored, `CORRUPTED_IMAGE` or
                `UNSUPPORTED_FORMAT` if it cannot be decoded, `MODEL_INFERENCE_FAILED` if the
                model fails.
        """
        raw = self.storage.load(image_sha256)
        if raw is None:
            raise LoadvisionError(ErrorCode.ROUTE_NOT_FOUND, f"photo {image_sha256} is not stored")
        image = prepare_image(raw, self.limits)
        try:
            return explainer.explain(image, kind)
        except Exception as error:
            raise LoadvisionError(
                ErrorCode.MODEL_INFERENCE_FAILED, f"{type(error).__name__}: {error}"
            ) from error
