import logging
import os
import sys
import google.auth
from google.adk.telemetry.google_cloud import get_gcp_exporters, get_gcp_resource
from google.adk.telemetry.setup import maybe_set_otel_providers
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk._logs import LoggingHandler

logger = logging.getLogger(__name__)

def cloud_telemetry_enabled() -> bool:
    return os.getenv("ENABLE_CLOUD_TELEMETRY","").strip().lower() in ("1","true","yes")

def configure_root_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(stdout)
    root.addHandler(LoggingHandler())

def setup_telemetry(service_name: str) -> None:
    if not cloud_telemetry_enabled():
        return

    os.environ.setdefault("OTEL_SERVICE_NAME",service_name)

    credentials,project_id = google.auth.default()
    hooks = get_gcp_exporters(
        enable_cloud_tracing=True,
        enable_cloud_metrics=True,
        enable_cloud_logging=True,
        google_auth=(credentials,project_id)
    )
    maybe_set_otel_providers([hooks],get_gcp_resource(project_id))
    configure_root_logging()

    HTTPXClientInstrumentor().instrument()

    logger.info("Cloud telemetry enabled for %s",service_name)

def instrument_asgi(app):

    if not cloud_telemetry_enabled():
        return app

    return OpenTelemetryMiddleware(app)
