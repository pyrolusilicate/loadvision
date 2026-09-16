"""Catalog of error codes shared by the API, the pipeline and logs."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Self


class ErrorCode(StrEnum):
    """Stable error codes. XML responses, the XSD and log events use these exact strings."""

    INVALID_CONTENT_TYPE = "INVALID_CONTENT_TYPE"
    INVALID_XML = "INVALID_XML"
    INVALID_SHIPMENT_ID = "INVALID_SHIPMENT_ID"
    INVALID_ENCODING = "INVALID_ENCODING"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    CORRUPTED_IMAGE = "CORRUPTED_IMAGE"
    SHIPMENT_NOT_FOUND = "SHIPMENT_NOT_FOUND"
    ASSESSMENT_NOT_FOUND = "ASSESSMENT_NOT_FOUND"
    TRAINING_ALREADY_RUNNING = "TRAINING_ALREADY_RUNNING"
    MODEL_VERSION_STATUS_CONFLICT = "MODEL_VERSION_STATUS_CONFLICT"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    MODEL_INFERENCE_FAILED = "MODEL_INFERENCE_FAILED"
    STAGE_CHECK_FAILED = "STAGE_CHECK_FAILED"
    MODEL_PACKAGE_INVALID = "MODEL_PACKAGE_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    """How an error is reported to API clients.

    Attributes:
        http_status: Status of the HTTP response, or None for errors that never become one,
            such as pipeline stage failures.
        message: What went wrong, in Russian.
        hint: What the client can do about it, in Russian.
    """

    http_status: int | None
    message: str
    hint: str


# Texts carry no limit values or id formats: limits live in configs/app.yaml and the id format in
# the domain and the XSD, so numbers copied here would drift from them.
ERROR_CATALOG: Final[Mapping[ErrorCode, ErrorSpec]] = MappingProxyType(
    {
        ErrorCode.INVALID_CONTENT_TYPE: ErrorSpec(
            415,
            "Тело запроса должно быть XML.",
            "Передайте заголовок Content-Type: application/xml; charset=utf-8.",
        ),
        ErrorCode.INVALID_XML: ErrorSpec(
            400,
            "XML не удалось разобрать, или он не соответствует схеме.",
            "Проверьте документ по схеме GET /api/v1/schema.xsd. "
            "DOCTYPE и внешние сущности не принимаются.",
        ),
        ErrorCode.INVALID_SHIPMENT_ID: ErrorSpec(
            400,
            "Номер перевозки имеет неверный формат.",
            "Допустимый формат номера описан в схеме GET /api/v1/schema.xsd.",
        ),
        ErrorCode.INVALID_ENCODING: ErrorSpec(
            400,
            "Фото передано не в корректном base64.",
            "Закодируйте байты файла в base64 без префикса «data:image/...;base64,».",
        ),
        ErrorCode.PAYLOAD_TOO_LARGE: ErrorSpec(
            413,
            "Запрос или фото больше допустимого размера.",
            "Уменьшите разрешение или пересожмите фото в JPEG. Лимиты описаны в документации API.",
        ),
        ErrorCode.UNSUPPORTED_FORMAT: ErrorSpec(
            415,
            "Формат изображения не поддерживается.",
            "Отправьте JPEG, PNG или WebP. Фото в HEIC сначала сконвертируйте в JPEG.",
        ),
        ErrorCode.CORRUPTED_IMAGE: ErrorSpec(
            422,
            "Изображение не удалось декодировать: файл повреждён, обрезан "
            "или его размеры вне допустимых.",
            "Отправьте исходный файл фото без изменений. "
            "Требования к размерам описаны в документации API.",
        ),
        ErrorCode.SHIPMENT_NOT_FOUND: ErrorSpec(
            404,
            "Перевозка с таким номером не найдена.",
            "Проверьте номер. Перевозка появляется после первой оценки фото "
            "через POST /api/v1/assessments.",
        ),
        ErrorCode.ASSESSMENT_NOT_FOUND: ErrorSpec(
            404,
            "Оценка с таким идентификатором не найдена.",
            "Возьмите идентификатор из ответа на оценку фото или из истории перевозки.",
        ),
        ErrorCode.TRAINING_ALREADY_RUNNING: ErrorSpec(
            409,
            "Дообучение уже выполняется.",
            "Дождитесь завершения текущей задачи и запустите дообучение снова.",
        ),
        ErrorCode.MODEL_VERSION_STATUS_CONFLICT: ErrorSpec(
            409,
            "Версию модели нельзя перевести в этот статус.",
            "Проверьте текущий статус версии в GET /api/v1/models.",
        ),
        ErrorCode.MODEL_NOT_READY: ErrorSpec(
            503,
            "Модель ещё не загружена.",
            "Повторите запрос через несколько секунд. Готовность сервиса показывает GET /health.",
        ),
        ErrorCode.MODEL_INFERENCE_FAILED: ErrorSpec(
            500,
            "Модель не смогла обработать фото.",
            "Повторите запрос, повтор безопасен. "
            "Если ошибка не проходит, сообщите requestId из ответа.",
        ),
        ErrorCode.STAGE_CHECK_FAILED: ErrorSpec(
            None,
            "Постусловие этапа пайплайна не выполнено.",
            "Откройте summary.json и run.jsonl этого запуска: там этап и причина.",
        ),
        ErrorCode.MODEL_PACKAGE_INVALID: ErrorSpec(
            None,
            "Пакет модели повреждён или не соответствует схеме манифеста.",
            "Скачайте пакет заново или пересоберите его. Какой файл или поле не прошли проверку, "
            "написано в логе.",
        ),
        ErrorCode.INTERNAL_ERROR: ErrorSpec(
            500,
            "Внутренняя ошибка сервиса.",
            "Повторите запрос позже. Если ошибка не проходит, сообщите requestId из ответа.",
        ),
    }
)


class LoadvisionError(Exception):
    """Error identified by a catalog code; the API turns it into an XML error response.

    Attributes:
        code: Catalog code of the error.
        detail: Context for logs only, never sent to clients. Must not contain image bytes,
            base64 data or secrets.
    """

    def __init__(self, code: ErrorCode, detail: str | None = None) -> None:
        super().__init__(code.value if detail is None else f"{code.value}: {detail}")
        self.code = code
        self.detail = detail

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        # Default exception pickling calls the class with self.args, the formatted message,
        # so an error sent back from a worker process would fail to unpickle. A subclass with
        # other constructor arguments overrides this method.
        return (type(self), (self.code, self.detail))

    @property
    def spec(self) -> ErrorSpec:
        """Catalog entry of this error's code."""
        return ERROR_CATALOG[self.code]
