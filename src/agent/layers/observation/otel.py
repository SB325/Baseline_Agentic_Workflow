from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
SystemMetricsInstrumentor().instrument() # Automatically collects CPU, memory, etc.

import time
import os   
import pdb
import subprocess
# Traces
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
# Metrics
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExportResult,
    PeriodicExportingMetricReader,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter # For OTLP/gRPC
# Logs
import logging
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from dotenv import load_dotenv
from enum import Enum
from uuid import uuid4

load_dotenv(override=True) 
in_docker = os.getenv("INDOCKER")
service_name = os.getenv("OTEL_SERVICE_NAME")
otelcol_service_name = os.getenv("OTEL_SERVICE_NAME")

def get_otel_ip():
    if bool(in_docker):
        return f'{otelcol_service_name}:4317'

    # Run a command and capture its stdout and stderr
    ip = subprocess.run(
        f"docker inspect --format='{{.NetworkSettings.Networks.homeserver.IPAddress}}' {otelcol_service_name}",
        capture_output=True,  # Capture stdout and stderr
        text=True,           # Decode output as text (UTF-8 by default)
        shell=True           # Raise CalledProcessError if the command returns a non-zero exit code
    ).stdout.replace('\n', '')
    return f'{ip}:4317'

class otel_tracer():
    # Configure the service name and other resource attributes
    resourceAttributes = Resource(attributes={
        "service.name": service_name,
    })
    # Set the global tracer provider
    provider = TracerProvider(resource=resourceAttributes)
    # Configure the OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=f"{get_otel_ip()}", insecure=True) 
    # Add a span processor to the provider
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(provider)
    # Acquire a tracer for use in your application
    tracer = trace.get_tracer(__name__)

    def set_span(self, span_name: str):
        print(f"Tracing span: {span_name}.")
        return self.tracer.start_as_current_span(span_name)

# ===============================
class meterType(Enum):
    counter = "AsynchronousCounter"
    upDownCounter = "AsynchronousUpDownCounter"
    gauge = "ObservableGauge"
    histogram = "Histogram"

# https://opentelemetry-python.readthedocs.io/en/latest/sdk/metrics.export.html#
# 1. Configure the MeterProvider with a Console Exporter (not necessary)
# The MetricExporter prints metrics to the console for demonstration purposes.
metric_reader = PeriodicExportingMetricReader(
        exporter=OTLPMetricExporter(endpoint=f"{get_otel_ip()}", insecure=True, timeout=30000),
        export_interval_millis=10000)
meter_provider = MeterProvider(metric_readers=[metric_reader])

# Sets the global default meter provider
metrics.set_meter_provider(meter_provider)
class otel_metrics():
    meter_objects = {
                'counters': [], 
                'upDownCounters': [], 
                'observableGauges': [],
                'histograms': [],
            }
    
    def create_meter(self, 
            meter_name : str, 
            callbacks: None,
            meter_type: meterType,
            description: str,
            ):
        # 2. Acquire a meter
        # A meter is associated with a specific name (e.g., application/library name) and version.
        meter = metrics.get_meter(name=meter_name, version="1.39.1")

        try:
            meterType(meter_type)
        except ValueError as e:
            print(f"Invalid meter name: {e}")

        if meter_type == meterType.counter.value:
            # 3. Create a synchronous instrument (Counter)
            # Counters are used for values that only increase, like the number of requests or errors.
            self.meter_objects['counters'].append(
                meter.create_counter(
                    meter_name,  # ex: "http.server.requests.total",
                    unit="1",
                    description=description,
                )
            )
        if meter_type == meterType.upDownCounter.value:
            self.meter_objects['upDownCounters'].append(
                meter.create_up_down_counter(
                    meter_name,  # ex: "http.server.requests.total",
                    unit="1",
                    description=description,
                )
            )
            
        if meter_type == meterType.gauge.value:
            self.meter_objects['observableGauges'].append(
                meter.create_observable_gauge(
                    meter_name,  # ex: "http.server.requests.total",
                    [callbacks],
                    unit="1",
                    description=description,
                )
            )
        if meter_type == meterType.histogram.value:
            self.meter_objects['histograms'].append(
                meter.create_histogram(
                    meter_name,  # ex: "http.server.requests.total",
                    unit="1",
                    description=description,
                )
            )
    
    def update_counter(self, counter_name, increase_by: int = 1, attributes: dict = None):
        counter = [val for val in self.meter_objects['counters']
                if counter_name in val.name]
        if not counter:
            print(f"Counter name {counter_name} not found.")

        counter[0].add(increase_by, attributes)

    def update_up_down_counter(self, counter_name, change_by: int = 1, attributes: dict = None):
        upDownCounter = [val for val in self.meter_objects['upDownCounters']
                if counter_name in val.name]

        if not upDownCounter:
            print(f"UpDownCounter name {counter_name} not found.")
        # pdb.set_trace()
        upDownCounter[0].add(change_by, attributes)

    def update_gauge(self,):
        gauge = [val for val in self.meter_objects['gauges']
                if counter_name in val.name]
    
        if not gauge:
            print(f"Gauge name {counter_name} not found.")

    def update_histogram(self, counter_name, ammount: int, attributes: dict=None, context=None):
        histogram = [val for val in self.meter_objects['histograms']
                if counter_name in val.name]
        
        if not histogram:
            print(f"Histogram name {counter_name} not found.")
        
        histogram[0].record(ammount, attributes, context)

# =======================================================================
class otel_logger():
    resource = Resource.create({ "service.instance.id": 'market_reader'})
    # 2. Configure the Logger Provider
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)
    # 3. Configure the exporter (e.g., OTLP to a collector running on localhost:4317)
    # Use insecure=True if your collector does not have TLS configured
    log_exporter = OTLPLogExporter(endpoint=f"{get_otel_ip()}", insecure=True)
    # 4. Add a processor to the provider
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    # 5. Attach OTLP handler to the Python standard logging
    handler = LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider)
    logging.getLogger().setLevel(logging.DEBUG) # Ensure all levels are captured
    logging.getLogger().addHandler(handler)
    # 6. Use standard Python logging
    logger = logging.getLogger(__name__)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg, exc_info=True)
    
    def exception(self, msg):
        self.logger.exception(msg)


if __name__ == "__main__":
    metric = otel_metrics()
    traces = otel_tracer()
    logs = otel_logger()

    metric.create_meter(
            meter_name = 'newmeter.test', 
            meter_type = "AsynchronousCounter",
            description = "Test async counter meter"
        )
    
    for i in range(5):
        metric.update_counter(
                counter_name = 'newmeter.test', 
                attributes = {'attribute': 1234})

    with traces.set_span(span_name = "First Span"):
        logs.info(f'My First Span')
        with traces.set_span(span_name = "Second Span"):
            logs.info(f'My Second Span')
    
    print('done')
