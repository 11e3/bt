"""Metrics exporters for external monitoring systems.

Provides Prometheus-style metrics export and integration
with external monitoring dashboards.
"""

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from bt.monitoring.metrics import MetricsCollector, MetricValue


@dataclass
class PrometheusMetric:
    """Represents a Prometheus-style metric."""

    name: str
    type_: str  # gauge, counter, histogram
    help_text: str
    labels: dict[str, str]
    value: float
    timestamp: float | None = None


class PrometheusExporter:
    """Exports metrics in Prometheus format."""

    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector

    def export_metrics(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []

        # Group metrics by name
        metrics_by_name = {}
        for metric_list in self.metrics_collector._metrics.values():
            for metric in metric_list:
                if metric.name not in metrics_by_name:
                    metrics_by_name[metric.name] = []
                metrics_by_name[metric.name].append(metric)

        # Export each metric family
        for metric_name, metrics in metrics_by_name.items():
            if not metrics:
                continue

            # Determine metric type from the metric name/unit
            metric_type = self._determine_metric_type(metric_name, metrics[0])

            # Add HELP comment
            help_text = self._get_metric_help(metric_name)
            lines.append(f"# HELP {metric_name} {help_text}")

            # Add TYPE comment
            lines.append(f"# TYPE {metric_name} {metric_type}")

            # Add metric values
            for metric in metrics[-10:]:  # Export last 10 values
                labels_str = ""
                if metric.labels:
                    labels_parts = [f'{k}="{v}"' for k, v in metric.labels.items()]
                    labels_str = f"{{{','.join(labels_parts)}}}"

                timestamp_str = ""
                if metric.timestamp:
                    timestamp_str = f" {int(metric.timestamp.timestamp() * 1000)}"

                lines.append(f"{metric_name}{labels_str} {metric.value}{timestamp_str}")

            lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def _determine_metric_type(self, name: str, sample_metric: MetricValue) -> str:
        """Determine Prometheus metric type."""
        if "counter" in name.lower() or sample_metric.unit == "counter":
            return "counter"
        if "histogram" in name.lower() or sample_metric.unit == "histogram":
            return "histogram"
        return "gauge"

    def _get_metric_help(self, name: str) -> str:
        """Get help text for a metric."""
        help_texts = {
            "backtest.duration": "Time taken to execute backtest",
            "backtest.symbols_count": "Number of symbols in backtest",
            "backtest.bars_count": "Total number of price bars processed",
            "backtest.trades_count": "Number of trades executed",
            "backtest.total_return": "Total portfolio return percentage",
            "backtest.bars_per_second": "Data processing rate",
            "backtest.trades_per_second": "Trade execution rate",
            "system.cpu_percent": "System CPU usage percentage",
            "system.memory_percent": "System memory usage percentage",
            "system.memory_used": "System memory used in MB",
            "process.memory_rss": "Process RSS memory in MB",
            "process.cpu_percent": "Process CPU usage percentage",
            "process.uptime": "Process uptime in seconds",
            "function.duration": "Function execution time",
            "operation.duration": "Operation execution time",
        }

        return help_texts.get(name, f"BT Framework metric: {name}")


class JSONMetricsExporter:
    """Exports metrics in JSON format for custom dashboards."""

    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector

    def export_metrics(self) -> str:
        """Export all metrics as JSON."""
        all_metrics = []

        for metric_list in self.metrics_collector._metrics.values():
            for metric in metric_list:
                metric_dict = {
                    "name": metric.name,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "labels": metric.labels,
                    "unit": metric.unit,
                }
                all_metrics.append(metric_dict)

        # Add gauge and counter metrics
        for name, value in self.metrics_collector._gauges.items():
            all_metrics.append(
                {
                    "name": name,
                    "value": value,
                    "timestamp": None,
                    "labels": {},
                    "unit": "gauge",
                }
            )

        for name, value in self.metrics_collector._counters.items():
            all_metrics.append(
                {
                    "name": name,
                    "value": value,
                    "timestamp": None,
                    "labels": {},
                    "unit": "counter",
                }
            )

        return json.dumps(
            {"timestamp": time.time(), "metrics": all_metrics, "summary": self._generate_summary()},
            indent=2,
        )

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        total_metrics = sum(len(lst) for lst in self.metrics_collector._metrics.values())

        return {
            "total_metrics": total_metrics,
            "metric_families": len(self.metrics_collector._metrics),
            "gauges": len(self.metrics_collector._gauges),
            "counters": len(self.metrics_collector._counters),
            "histograms": len(self.metrics_collector._histograms),
            "alert_rules": len(self.metrics_collector._alert_rules),
        }


class MetricsHTTPServer(BaseHTTPRequestHandler):
    """HTTP server for serving metrics endpoints."""

    def __init__(self, metrics_collector, *args, **kwargs):
        self.metrics_collector = metrics_collector
        self.prometheus_exporter = PrometheusExporter(metrics_collector)
        self.json_exporter = JSONMetricsExporter(metrics_collector)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/metrics":
            self._serve_prometheus_metrics()
        elif self.path == "/metrics/json":
            self._serve_json_metrics()
        elif self.path == "/health":
            self._serve_health_check()
        else:
            self.send_error(404, "Endpoint not found")

    def _serve_prometheus_metrics(self):
        """Serve metrics in Prometheus format."""
        try:
            metrics_text = self.prometheus_exporter.export_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(metrics_text)))
            self.end_headers()
            self.wfile.write(metrics_text.encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"Error generating metrics: {str(e)}")

    def _serve_json_metrics(self):
        """Serve metrics in JSON format."""
        try:
            metrics_json = self.json_exporter.export_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(metrics_json)))
            self.end_headers()
            self.wfile.write(metrics_json.encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"Error generating JSON metrics: {str(e)}")

    def _serve_health_check(self):
        """Serve health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        health_data = json.dumps(
            {
                "status": "healthy",
                "timestamp": time.time(),
                "metrics_collector": "active",
                "total_metrics": sum(len(lst) for lst in self.metrics_collector._metrics.values()),
            }
        )
        self.send_header("Content-Length", str(len(health_data)))
        self.end_headers()
        self.wfile.write(health_data.encode("utf-8"))

    def log_message(self, format, *args):
        """Override to use structured logging."""
        from bt.monitoring import get_structured_logger

        logger = get_structured_logger()
        logger.logger.info(
            f"HTTP {format % args}",
            extra={
                "http_method": self.command,
                "http_path": self.path,
                "client_ip": self.client_address[0] if self.client_address else None,
            },
        )


class MetricsServer:
    """HTTP server for exposing metrics endpoints."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        metrics_collector: MetricsCollector | None = None,
    ):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        # Deferred import to avoid circular dependency
        if metrics_collector is None:
            from bt.monitoring import get_metrics_collector

            metrics_collector = get_metrics_collector()
        self.metrics_collector = metrics_collector

    def start(self):
        """Start the metrics server in a background thread."""

        def create_handler(*args, **kwargs):
            return MetricsHTTPServer(self.metrics_collector, *args, **kwargs)

        self.server = HTTPServer((self.host, self.port), create_handler)

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        from bt.monitoring import get_structured_logger

        logger = get_structured_logger()
        logger.logger.info(f"Metrics server started on http://{self.host}:{self.port}")

    def stop(self):
        """Stop the metrics server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

        if self.thread:
            self.thread.join(timeout=5.0)

        from bt.monitoring import get_structured_logger

        logger = get_structured_logger()
        logger.logger.info("Metrics server stopped")

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self.server is not None and self.thread is not None and self.thread.is_alive()


# Integration utilities


def setup_monitoring_integration():
    """Set up monitoring integration for the framework."""
    from bt.monitoring import get_performance_monitor

    # Start metrics collection in background
    monitor = get_performance_monitor()

    def collect_system_metrics():
        """Background task to collect system metrics."""
        import time

        while True:
            monitor.record_system_metrics()
            time.sleep(60)  # Collect every minute

    # Start background collection thread
    collection_thread = threading.Thread(target=collect_system_metrics, daemon=True)
    collection_thread.start()

    # Start metrics server
    metrics_server = MetricsServer()
    metrics_server.start()

    return {
        "metrics_server": metrics_server,
        "collection_thread": collection_thread,
    }


# Alerting integrations


class SlackAlerter:
    """Send alerts to Slack."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(self, alert_data: dict[str, Any]):
        """Send alert to Slack."""
        import requests

        message = {
            "text": f"🚨 BT Framework Alert: {alert_data['message']}",
            "attachments": [
                {
                    "color": self._get_color_for_severity(alert_data.get("severity", "info")),
                    "fields": [
                        {"title": "Rule", "value": alert_data["rule"], "short": True},
                        {"title": "Metric", "value": alert_data["metric"], "short": True},
                        {"title": "Value", "value": str(alert_data["value"]), "short": True},
                        {
                            "title": "Severity",
                            "value": alert_data.get("severity", "info"),
                            "short": True,
                        },
                    ],
                }
            ],
        }

        try:
            response = requests.post(self.webhook_url, json=message, timeout=10)
            response.raise_for_status()
        except Exception as e:
            from bt.monitoring import get_structured_logger

            logger = get_structured_logger()
            logger.logger.error(f"Failed to send Slack alert: {e}")

    def _get_color_for_severity(self, severity: str) -> str:
        """Get Slack color for severity level."""
        colors = {
            "info": "good",
            "warning": "warning",
            "error": "danger",
            "critical": "danger",
        }
        return colors.get(severity, "warning")


class EmailAlerter:
    """Send alerts via email."""

    def __init__(self, smtp_config: dict[str, Any]):
        self.smtp_config = smtp_config

    def send_alert(self, alert_data: dict[str, Any]):
        """Send alert via email."""
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart()
        msg["From"] = self.smtp_config["from_email"]
        msg["To"] = self.smtp_config["to_email"]
        msg["Subject"] = f"BT Framework Alert: {alert_data['rule']}"

        body = f"""
        BT Framework Alert Triggered

        Rule: {alert_data["rule"]}
        Metric: {alert_data["metric"]}
        Value: {alert_data["value"]}
        Severity: {alert_data.get("severity", "info")}
        Message: {alert_data["message"]}
        Timestamp: {alert_data["timestamp"]}
        """

        msg.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP(self.smtp_config["smtp_server"], self.smtp_config["smtp_port"])
            if self.smtp_config.get("use_tls"):
                server.starttls()
            if self.smtp_config.get("username"):
                server.login(self.smtp_config["username"], self.smtp_config["password"])

            server.send_message(msg)
            server.quit()

        except Exception as e:
            from bt.monitoring import get_structured_logger

            logger = get_structured_logger()
            logger.logger.error(f"Failed to send email alert: {e}")


# Global metrics server instance
_metrics_server = None


def start_metrics_server(host: str = "localhost", port: int = 8000) -> MetricsServer:
    """Start the global metrics server."""
    global _metrics_server

    if _metrics_server is None:
        _metrics_server = MetricsServer(host, port)
        _metrics_server.start()

    return _metrics_server


def stop_metrics_server():
    """Stop the global metrics server."""
    global _metrics_server

    if _metrics_server:
        _metrics_server.stop()
        _metrics_server = None
