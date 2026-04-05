import contextvars
import json
import logging
from datetime import datetime, timezone

REQUEST_ID_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id",
    default=None,
)


class JsonLogFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": getattr(record, "service", self._service),
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in (
            "event",
            "job_id",
            "request_id",
            "path",
            "method",
            "status_code",
            "duration_ms",
            "client_ip",
            "api_version",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            request_id = REQUEST_ID_CTX.get()
            if request_id:
                record.request_id = request_id
        return True


def configure_json_logging(service: str) -> None:
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter(service=service))
    handler.addFilter(RequestIdFilter())

    root.addHandler(handler)
    root.setLevel(logging.INFO)


def set_request_id(request_id: str) -> contextvars.Token[str | None]:
    return REQUEST_ID_CTX.set(request_id)


def reset_request_id(token: contextvars.Token[str | None]) -> None:
    REQUEST_ID_CTX.reset(token)

