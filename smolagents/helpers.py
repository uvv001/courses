import os
import base64
from smolagents import HfApiModel, LiteLLMModel
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

class ArgumentsHelper:
    def __init__(self):
        import argparse
        parser = argparse.ArgumentParser(description="Run the CodeAgent with a specified provider.")
        parser.add_argument("--provider", choices=["hg", "openrouter"], default="hg", help="Model provider: 'hg' (default) or 'openrouter'.")
        self.args = parser.parse_args()

    def getModel(self):
        if self.args.provider == "hg":
            return HfApiModel()
        
        if self.args.provider == "openrouter":
            return LiteLLMModel(
                model_id="openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
                api_key=os.environ["OPENROUTER_API_KEY"]
            )

def register_opentelemetry_through_langfuse():
    """
    Register the OpenTelemetry tracer with Langfuse.
    """
    LANGFUSE_AUTH=base64.b64encode(f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)